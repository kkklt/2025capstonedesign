#include <ros/ros.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <path/CheckCollision.h>
#include <mutex>

class OctomapChecker
{
public:
    OctomapChecker()
    {
        ros::NodeHandle nh;

        octree_ = nullptr;

        // Subscriber
        octomap_sub_ = nh.subscribe("octomap_binary", 1, &OctomapChecker::octomapCallback, this);

        // Service
        service_ = nh.advertiseService("check_collision_at_point", &OctomapChecker::handleCollisionCheck, this);

        ROS_INFO("Octomap checker service is ready.");
    }

private:
    std::shared_ptr<octomap::OcTree> octree_;
    ros::Subscriber octomap_sub_;
    ros::ServiceServer service_;
    std::mutex mtx_;

    void octomapCallback(const octomap_msgs::Octomap::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        try {
            octomap::AbstractOcTree* tree = octomap_msgs::binaryMsgToMap(*msg);
            if (tree) {
                // dynamic_cast to OcTree
                octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);
                if (octree) {
                    octree_.reset(octree);
                    ROS_INFO_ONCE("Successfully received and parsed first OctoMap message.");
                } else {
                    ROS_ERROR("Received Octomap is not an OcTree!");
                    delete tree;
                }
            }
        } catch (std::exception& e) {
            ROS_ERROR("Failed to parse Octomap message: %s", e.what());
        }
    }

    bool handleCollisionCheck(path::CheckCollision::Request& req,
                              path::CheckCollision::Response& res)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!octree_) {
            ROS_WARN_THROTTLE(5, "Octomap not yet received. Assuming collision for safety.");
            res.is_occupied = true;
            return true;
        }

        try {
            octomap::OcTreeNode* node = octree_->search(req.point.x, req.point.y, req.point.z);

            if (node && octree_->isNodeOccupied(node)) {
                res.is_occupied = true;
            } else {
                res.is_occupied = false;
            }

        } catch (std::exception& e) {
            ROS_ERROR("Error during Octomap search: %s", e.what());
            res.is_occupied = true;
        }

        return true;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "octomap_checker_node");

    OctomapChecker checker;

    ros::spin();
    return 0;
}