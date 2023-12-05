;; Auto-generated. Do not edit!


(when (boundp 'robot_controller::DetectObjects)
  (if (not (find-package "ROBOT_CONTROLLER"))
    (make-package "ROBOT_CONTROLLER"))
  (shadow 'DetectObjects (find-package "ROBOT_CONTROLLER")))
(unless (find-package "ROBOT_CONTROLLER::DETECTOBJECTS")
  (make-package "ROBOT_CONTROLLER::DETECTOBJECTS"))
(unless (find-package "ROBOT_CONTROLLER::DETECTOBJECTSREQUEST")
  (make-package "ROBOT_CONTROLLER::DETECTOBJECTSREQUEST"))
(unless (find-package "ROBOT_CONTROLLER::DETECTOBJECTSRESPONSE")
  (make-package "ROBOT_CONTROLLER::DETECTOBJECTSRESPONSE"))

(in-package "ROS")

(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))




(defclass robot_controller::DetectObjectsRequest
  :super ros::object
  :slots (_image ))

(defmethod robot_controller::DetectObjectsRequest
  (:init
   (&key
    ((:image __image) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _image __image)
   self)
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;;
   self)
  )

(defclass robot_controller::DetectObjectsResponse
  :super ros::object
  :slots (_object ))

(defmethod robot_controller::DetectObjectsResponse
  (:init
   (&key
    ((:object __object) (instance robot_controller::DetectedObject :init))
    )
   (send-super :init)
   (setq _object __object)
   self)
  (:object
   (&rest __object)
   (if (keywordp (car __object))
       (send* _object __object)
     (progn
       (if __object (setq _object (car __object)))
       _object)))
  (:serialization-length
   ()
   (+
    ;; robot_controller/DetectedObject _object
    (send _object :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; robot_controller/DetectedObject _object
       (send _object :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; robot_controller/DetectedObject _object
     (send _object :deserialize buf ptr-) (incf ptr- (send _object :serialization-length))
   ;;
   self)
  )

(defclass robot_controller::DetectObjects
  :super ros::object
  :slots ())

(setf (get robot_controller::DetectObjects :md5sum-) "66b6b06f35191dce919e07c706baac8b")
(setf (get robot_controller::DetectObjects :datatype-) "robot_controller/DetectObjects")
(setf (get robot_controller::DetectObjects :request) robot_controller::DetectObjectsRequest)
(setf (get robot_controller::DetectObjects :response) robot_controller::DetectObjectsResponse)

(defmethod robot_controller::DetectObjectsRequest
  (:response () (instance robot_controller::DetectObjectsResponse :init)))

(setf (get robot_controller::DetectObjectsRequest :md5sum-) "66b6b06f35191dce919e07c706baac8b")
(setf (get robot_controller::DetectObjectsRequest :datatype-) "robot_controller/DetectObjectsRequest")
(setf (get robot_controller::DetectObjectsRequest :definition-)
      "sensor_msgs/Image image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
DetectedObject object


================================================================================
MSG: robot_controller/DetectedObject
int32 x1
int32 y1
int32 x2
int32 y2
string class_name
")

(setf (get robot_controller::DetectObjectsResponse :md5sum-) "66b6b06f35191dce919e07c706baac8b")
(setf (get robot_controller::DetectObjectsResponse :datatype-) "robot_controller/DetectObjectsResponse")
(setf (get robot_controller::DetectObjectsResponse :definition-)
      "sensor_msgs/Image image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
DetectedObject object


================================================================================
MSG: robot_controller/DetectedObject
int32 x1
int32 y1
int32 x2
int32 y2
string class_name
")



(provide :robot_controller/DetectObjects "66b6b06f35191dce919e07c706baac8b")


