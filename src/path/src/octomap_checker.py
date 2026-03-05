import rospy
import octomap
import threading
from octomap_msgs.msg import Octomap
from capstone.srv import CheckCollision, CheckCollisionResponse

class OctomapChecker:
    def __init__(self):
        rospy.init_node('octomap_checker_node', anonymous=True)

        self.octrr = None
        self.lock = threading.Lock()

        self.octomap_sub = rospy.Subscriber('octomap_binary', Octomap, self.octomap_callback, queue_size=1)
        self.collision_service = rospy.Service('check_collision_at_point', CheckCollision, self.handle_collision_check)

        rospy.loginfo("Octomap checker service is ready.")

    def octomap_callback(self, msg):
        with self.lock:
            try:
                self.octree = octomap.OcTree.readBinary(msg.data)
                rospy.loginfo_once("Successfully received and parsed first OctoMap message.")
            except Exception as e:
                rospy.logerr("Failed to parse Octomap message: %s", e)
    
    def handle_collision_check(self, req):
        response = CheckCollisionResponse()

        with self.lock:
            if self.octrr is None:
                rospy.logwarn_throttle(5, "Octomap not yey received. Assuming collision for safety.")
                response.is_occupied = True
                return response
            
            point = (req.point.x, req.point.y, req.point.z)

            try:
                node = self.octree.search(point[0], point[1], point[2])

                if node and self.octree.isNodeOccupied(node):
                    response.is_occupied = True
                else:
                    response.is_occupied = False
            
            except Exception as e:
                rospy.logerr("Error during Octomap search: %s", e)
                
                response.is_occupied = True

        return response

if __name__ == '__main__':
    try :
        checker = OctomapChecker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass