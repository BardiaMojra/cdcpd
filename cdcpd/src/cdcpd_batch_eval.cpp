// cdcpd_batch_eval.cpp
//
// Standalone (no ROS node, no roscore) batch evaluation driver for CDCPD.
// Reads a directory of already-extracted DLO point clouds (this project's
// dlo_perception_v022.py's dlo_raw_dbscan_pcd/*.pcd -- DBSCAN-cleaned,
// background/pole/lee already removed, but NOT yet spline-fit or
// node-ordered) directly as CDCPD's per-frame observed cloud X, bypassing
// SegmenterHSV entirely (see cdcpd.h's track_preextracted_cloud() doc
// comment for why that method exists instead of reusing the normal
// points-based operator()).
//
// ros::init() is called once below purely because CDCPD's constructor
// still stores a ros::NodeHandle member (nothing else in this file touches
// ROS) -- confirmed via direct test that ros::init() never contacts or
// requires a reachable master; only live publish/subscribe/param calls
// would. No roscore process is needed to run this executable.
//
// Output: one CSV row per frame (frame_no,status,proc_time_s,node_count,
// dlo_d01_x,dlo_d01_y,dlo_d01_z,...) written directly -- kept deliberately
// simple/raw here; the Python wrapper (eval_perception_noros.py) turns this
// into the full run_frame_log.csv/run_err_log.txt/episode_summary.log
// report format, matching eval_perception.py's existing output.
//
// Usage:
//   cdcpd_batch_eval --input_dir <dlo_raw_dbscan_pcd dir> --output_csv <path>
//                     [--num_points N] [--max_rope_length M] [--initial_rope_z Z]
//                     [--init_start X,Y,Z] [--init_end X,Y,Z]
//                     [--endpoints_csv PATH]   (per-frame pole+gripper fixed points)
//                     [--alpha A] [--beta B] [--lambda L] [--k K] [--zeta Z]
//                     [--obstacle_cost_weight W] [--fixed_points_weight W]
//                     [--objective_value_threshold T] [--use_recovery 0|1]
//                     [--max_frames N]

#include <ros/ros.h>

#include "cdcpd/cdcpd.h"
#include "cdcpd/deformable_object_configuration.h"

#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Args {
    std::string input_dir;
    std::string output_csv;
    int num_points = 25;
    float max_rope_length = 0.8f;
    float initial_rope_z = 1.0f;
    // Explicit initial-template endpoints, in the SAME frame as the input clouds.
    // When unset, the template falls back to the camera-frame straight line
    // cdcpd_node.cpp uses for the no-gripper case (see main()). That fallback is wrong
    // for any input that is not in camera frame: this harness feeds robot-frame clouds
    // (dlo_perception applies lee_H before saving dlo_raw_crop_pcd), so the default
    // template started ~1.5 m away from the data and CDCPD spent the whole episode
    // dragging itself toward the rope -- 0.72 m mean error on a 0.65 m rope, decaying
    // monotonically frame over frame instead of tracking.
    bool have_init_endpoints = false;
    Eigen::Vector3f init_start = Eigen::Vector3f::Zero();
    Eigen::Vector3f init_end = Eigen::Vector3f::Zero();
    // Per-frame known endpoints ("w/ gripper" evaluation variant): frame index (1-based, this
    // program's own input order) -> (pole/anchor, gripper), same frame as the input clouds.
    // Written by perc_variants.write_endpoints_csv(); passed to CDCPD as two fixed points.
    std::string endpoints_csv;
    double alpha = 0.5;
    double beta = 1.0;
    double lambda = 1.0;
    double k = 100.0;
    double zeta = 10.0;
    double obstacle_cost_weight = 0.001;
    double fixed_points_weight = 10.0;
    double objective_value_threshold = 1.0;
    bool use_recovery = false;
    int max_frames = -1;  // -1 = all frames in input_dir
};

static void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --input_dir DIR --output_csv PATH"
                 " [--num_points N] [--max_rope_length M] [--initial_rope_z Z]"
                 " [--init_start X,Y,Z] [--init_end X,Y,Z] [--endpoints_csv PATH]"
                 " [--alpha A] [--beta B] [--lambda L] [--k K] [--zeta Z]"
                 " [--obstacle_cost_weight W] [--fixed_points_weight W]"
                 " [--objective_value_threshold T] [--use_recovery 0|1]"
                 " [--max_frames N]\n";
}

static Args parse_args(int argc, char **argv) {
    Args a;
    auto need_value = [&](int i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << argv[i] << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
        return std::string(argv[i + 1]);
    };
    auto parse_vec3 = [&](const std::string &text, const char *flag) -> Eigen::Vector3f {
        Eigen::Vector3f v;
        std::string s2 = text;
        std::replace(s2.begin(), s2.end(), ',', ' ');
        std::istringstream iss(s2);
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            std::cerr << "bad value for " << flag << " (want X,Y,Z): " << text << "\n";
            std::exit(2);
        }
        return v;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--input_dir") { a.input_dir = need_value(i); ++i; }
        else if (arg == "--init_start") {
            a.init_start = parse_vec3(need_value(i), "--init_start");
            a.have_init_endpoints = true; ++i;
        }
        else if (arg == "--init_end") {
            a.init_end = parse_vec3(need_value(i), "--init_end");
            a.have_init_endpoints = true; ++i;
        }
        else if (arg == "--endpoints_csv") {
            a.endpoints_csv = need_value(i); ++i;
        }
        else if (arg == "--output_csv") { a.output_csv = need_value(i); ++i; }
        else if (arg == "--num_points") { a.num_points = std::stoi(need_value(i)); ++i; }
        else if (arg == "--max_rope_length") { a.max_rope_length = std::stof(need_value(i)); ++i; }
        else if (arg == "--initial_rope_z") { a.initial_rope_z = std::stof(need_value(i)); ++i; }
        else if (arg == "--alpha") { a.alpha = std::stod(need_value(i)); ++i; }
        else if (arg == "--beta") { a.beta = std::stod(need_value(i)); ++i; }
        else if (arg == "--lambda") { a.lambda = std::stod(need_value(i)); ++i; }
        else if (arg == "--k") { a.k = std::stod(need_value(i)); ++i; }
        else if (arg == "--zeta") { a.zeta = std::stod(need_value(i)); ++i; }
        else if (arg == "--obstacle_cost_weight") { a.obstacle_cost_weight = std::stod(need_value(i)); ++i; }
        else if (arg == "--fixed_points_weight") { a.fixed_points_weight = std::stod(need_value(i)); ++i; }
        else if (arg == "--objective_value_threshold") { a.objective_value_threshold = std::stod(need_value(i)); ++i; }
        else if (arg == "--use_recovery") { a.use_recovery = std::stoi(need_value(i)) != 0; ++i; }
        else if (arg == "--max_frames") { a.max_frames = std::stoi(need_value(i)); ++i; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else { std::cerr << "unknown arg: " << arg << "\n"; print_usage(argv[0]); std::exit(2); }
    }
    if (a.input_dir.empty() || a.output_csv.empty()) {
        std::cerr << "--input_dir and --output_csv are required\n";
        print_usage(argv[0]);
        std::exit(2);
    }
    return a;
}

// dlo_perception_v022.py names dlo_raw_dbscan_pcd frames "NNNNNN_seq-...pcd" (zero-padded
// frame index prefix), so plain lexicographic sort already sorts them in frame order --
// no natural-sort needed.
static std::vector<std::string> list_pcd_files_sorted(const std::string &dir) {
    std::vector<std::string> files;
    DIR *d = opendir(dir.c_str());
    if (!d) {
        std::cerr << "ERROR: cannot open input_dir: " << dir << "\n";
        return files;
    }
    struct dirent *entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".pcd") {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

// frame_no,anchor_x,anchor_y,anchor_z,gripper_x,gripper_y,gripper_z -- the same format
// trackdlo_batch_eval.cpp reads. Missing frames simply get no fixed points.
static std::map<int, std::pair<Eigen::Vector3d, Eigen::Vector3d>>
load_endpoints_csv(const std::string &path) {
    std::map<int, std::pair<Eigen::Vector3d, Eigen::Vector3d>> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open --endpoints_csv: " << path << "\n";
        return out;
    }
    std::string line;
    std::getline(f, line);                 // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int frame_no; Eigen::Vector3d a, g;
        if (iss >> frame_no >> a.x() >> a.y() >> a.z() >> g.x() >> g.y() >> g.z()) {
            out[frame_no] = {a, g};
        }
    }
    std::cerr << "[cdcpd_batch_eval] " << out.size()
              << " frame(s) of known endpoints from " << path << "\n";
    return out;
}

int main(int argc, char **argv) {
    // See file header comment: never contacts/requires a reachable master, only needed
    // because CDCPD's constructor stores a ros::NodeHandle member.
    ros::init(argc, argv, "cdcpd_batch_eval",
              ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);

    Args args = parse_args(argc, argv);

    std::vector<std::string> pcd_files = list_pcd_files_sorted(args.input_dir);
    if (pcd_files.empty()) {
        std::cerr << "ERROR: no .pcd files found in " << args.input_dir << "\n";
        return 1;
    }
    if (args.max_frames > 0 && static_cast<int>(pcd_files.size()) > args.max_frames) {
        pcd_files.resize(args.max_frames);
    }
    std::cerr << "[cdcpd_batch_eval] " << pcd_files.size() << " frame(s) from " << args.input_dir << "\n";

    // Initial template: same straight-line-between-two-points construction cdcpd_node.cpp
    // uses for the no-gripper case (gripper_count == 0) -- see initialize_deformable_object_
    // configuration()'s call site in cdcpd_node.cpp.
    Eigen::Vector3f start_position(-args.max_rope_length / 2.0f, 0.0f, args.initial_rope_z);
    Eigen::Vector3f end_position(args.max_rope_length / 2.0f, 0.0f, args.initial_rope_z);
    if (args.have_init_endpoints) {
        start_position = args.init_start;
        end_position = args.init_end;
    }
    std::cerr << "[cdcpd_batch_eval] initial template: ("
              << start_position.x() << ", " << start_position.y() << ", " << start_position.z()
              << ") -> (" << end_position.x() << ", " << end_position.y() << ", "
              << end_position.z() << ")"
              << (args.have_init_endpoints ? " [from --init_start/--init_end]"
                                           : " [default camera-frame fallback]") << "\n";
    RopeConfiguration rope_config(args.num_points, args.max_rope_length, start_position, end_position);
    rope_config.initializeTracking();

    // NOTE: this constructor call requires ROS_MASTER_URI to point at a reachable master (even
    // though this executable never publishes/subscribes/reads params itself -- see this file's
    // header comment for what's been ruled out, and eval_perception_noros.py's config comment
    // for why: something deeper in CDCPD's dependency chain, not cdcpd.cpp itself, attempts a
    // live ros::Publisher registration during construction).
    CDCPD cdcpd(rope_config.initial_.points_, rope_config.initial_.edges_,
                args.objective_value_threshold, args.use_recovery, args.alpha, args.beta,
                args.lambda, args.k, args.zeta, args.obstacle_cost_weight, args.fixed_points_weight);

    PointCloud::Ptr current_template = rope_config.initial_.points_;

    std::map<int, std::pair<Eigen::Vector3d, Eigen::Vector3d>> endpoints;
    if (!args.endpoints_csv.empty()) {
        endpoints = load_endpoints_csv(args.endpoints_csv);
        if (endpoints.empty()) {
            std::cerr << "ERROR: --endpoints_csv given but no usable rows\n";
            return 1;
        }
    }
    // which template end is the pole: decided once, at the first constrained frame, by
    // which end sits nearer the given pole position -- a flipped template must never have
    // its gripper end pinned to the pole
    int pole_idx = -1, grip_idx = -1, n_constrained = 0;
    ObstacleConstraints no_obstacles;  // moveit disabled equivalent -- always empty

    std::ofstream out(args.output_csv);
    if (!out.is_open()) {
        std::cerr << "ERROR: cannot open output_csv for writing: " << args.output_csv << "\n";
        return 1;
    }
    out << "frame_no,status,proc_time_s,node_count";
    for (int i = 1; i <= args.num_points; ++i) {
        out << ",dlo_d" << std::setfill('0') << std::setw(2) << i << "_x"
            << ",dlo_d" << std::setfill('0') << std::setw(2) << i << "_y"
            << ",dlo_d" << std::setfill('0') << std::setw(2) << i << "_z";
    }
    out << "\n";
    out << std::setprecision(9);

    int n_fail = 0;
    for (size_t i = 0; i < pcd_files.size(); ++i) {
        int frame_no = static_cast<int>(i) + 1;
        PointCloud::Ptr cloud(new PointCloud);
        auto t0 = std::chrono::steady_clock::now();

        std::string status = "pass";
        int node_count = 0;
        Eigen::Matrix3Xf node_xyz;

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_files[i], *cloud) != 0) {
            std::cerr << "  [" << frame_no << "/" << pcd_files.size() << "] ERR_PCD_LOAD_FAILED: "
                      << pcd_files[i] << "\n";
            status = "fail";
            n_fail++;
        } else {
            smmap::AllGrippersSinglePose q_config;
            Eigen::MatrixXi gripper_idx;
            auto ep = endpoints.find(frame_no);
            if (ep != endpoints.end()) {
                if (pole_idx < 0) {
                    const Eigen::Matrix3Xf Yt = current_template->getMatrixXfMap().topRows(3);
                    const Eigen::Vector3f a = ep->second.first.cast<float>();
                    const bool first_is_pole = (Yt.col(0) - a).norm() <=
                                               (Yt.col(Yt.cols() - 1) - a).norm();
                    pole_idx = first_is_pole ? 0 : static_cast<int>(Yt.cols()) - 1;
                    grip_idx = first_is_pole ? static_cast<int>(Yt.cols()) - 1 : 0;
                    std::cerr << "[cdcpd_batch_eval] fixed points: pole -> node " << pole_idx
                              << ", gripper -> node " << grip_idx << "\n";
                }
                Eigen::Isometry3d pa = Eigen::Isometry3d::Identity();
                Eigen::Isometry3d pg = Eigen::Isometry3d::Identity();
                pa.translation() = ep->second.first;
                pg.translation() = ep->second.second;
                q_config.push_back(pa);
                q_config.push_back(pg);
                gripper_idx.resize(1, 2);
                gripper_idx << pole_idx, grip_idx;
                ++n_constrained;
            }
            auto result = cdcpd.track_preextracted_cloud(cloud, current_template, no_obstacles,
                    rope_config.max_segment_length_, {}, q_config, gripper_idx, 0);
            current_template = result.gurobi_output;  // recursive update, matches points_callback()

            if (result.status == OutputStatus::Success) {
                status = "pass";
            } else if (result.status == OutputStatus::ObjectiveTooHigh) {
                status = "warn";
            } else {
                status = "fail";
                n_fail++;
            }
            node_xyz = result.gurobi_output->getMatrixXfMap().topRows(3);
            node_count = static_cast<int>(result.gurobi_output->width);
        }

        auto t1 = std::chrono::steady_clock::now();
        double proc_time_s = std::chrono::duration<double>(t1 - t0).count();

        out << frame_no << "," << status << "," << proc_time_s << "," << node_count;
        for (int n = 0; n < args.num_points; ++n) {
            if (n < node_xyz.cols()) {
                out << "," << node_xyz(0, n) << "," << node_xyz(1, n) << "," << node_xyz(2, n);
            } else {
                out << ",nan,nan,nan";
            }
        }
        out << "\n";
        out.flush();

        if (frame_no % 20 == 0 || frame_no == static_cast<int>(pcd_files.size()) || status != "pass") {
            std::cerr << "  [" << frame_no << "/" << pcd_files.size() << "] status=" << status
                      << " node_count=" << node_count << " proc_time=" << proc_time_s << "s\n";
        }
    }

    out.close();
    std::cerr << "[cdcpd_batch_eval] done: " << pcd_files.size() << " frame(s), " << n_fail << " fail(s)";
    if (!args.endpoints_csv.empty()) {
        std::cerr << ", " << n_constrained << " frame(s) with fixed endpoints";
    }
    std::cerr << "\n";
    return n_fail > 0 ? 2 : 0;
}
