; Auto-generated. Do not edit!


(cl:in-package robot_controller-srv)


;//! \htmlinclude DetectObjects-request.msg.html

(cl:defclass <DetectObjects-request> (roslisp-msg-protocol:ros-message)
  ((image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass DetectObjects-request (<DetectObjects-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectObjects-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectObjects-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robot_controller-srv:<DetectObjects-request> is deprecated: use robot_controller-srv:DetectObjects-request instead.")))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <DetectObjects-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_controller-srv:image-val is deprecated.  Use robot_controller-srv:image instead.")
  (image m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectObjects-request>) ostream)
  "Serializes a message object of type '<DetectObjects-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectObjects-request>) istream)
  "Deserializes a message object of type '<DetectObjects-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectObjects-request>)))
  "Returns string type for a service object of type '<DetectObjects-request>"
  "robot_controller/DetectObjectsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObjects-request)))
  "Returns string type for a service object of type 'DetectObjects-request"
  "robot_controller/DetectObjectsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectObjects-request>)))
  "Returns md5sum for a message object of type '<DetectObjects-request>"
  "66b6b06f35191dce919e07c706baac8b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectObjects-request)))
  "Returns md5sum for a message object of type 'DetectObjects-request"
  "66b6b06f35191dce919e07c706baac8b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectObjects-request>)))
  "Returns full string definition for message of type '<DetectObjects-request>"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectObjects-request)))
  "Returns full string definition for message of type 'DetectObjects-request"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectObjects-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectObjects-request>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectObjects-request
    (cl:cons ':image (image msg))
))
;//! \htmlinclude DetectObjects-response.msg.html

(cl:defclass <DetectObjects-response> (roslisp-msg-protocol:ros-message)
  ((object
    :reader object
    :initarg :object
    :type robot_controller-msg:DetectedObject
    :initform (cl:make-instance 'robot_controller-msg:DetectedObject)))
)

(cl:defclass DetectObjects-response (<DetectObjects-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectObjects-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectObjects-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robot_controller-srv:<DetectObjects-response> is deprecated: use robot_controller-srv:DetectObjects-response instead.")))

(cl:ensure-generic-function 'object-val :lambda-list '(m))
(cl:defmethod object-val ((m <DetectObjects-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robot_controller-srv:object-val is deprecated.  Use robot_controller-srv:object instead.")
  (object m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectObjects-response>) ostream)
  "Serializes a message object of type '<DetectObjects-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'object) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectObjects-response>) istream)
  "Deserializes a message object of type '<DetectObjects-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'object) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectObjects-response>)))
  "Returns string type for a service object of type '<DetectObjects-response>"
  "robot_controller/DetectObjectsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObjects-response)))
  "Returns string type for a service object of type 'DetectObjects-response"
  "robot_controller/DetectObjectsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectObjects-response>)))
  "Returns md5sum for a message object of type '<DetectObjects-response>"
  "66b6b06f35191dce919e07c706baac8b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectObjects-response)))
  "Returns md5sum for a message object of type 'DetectObjects-response"
  "66b6b06f35191dce919e07c706baac8b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectObjects-response>)))
  "Returns full string definition for message of type '<DetectObjects-response>"
  (cl:format cl:nil "DetectedObject object~%~%~%================================================================================~%MSG: robot_controller/DetectedObject~%int32 x1~%int32 y1~%int32 x2~%int32 y2~%string class_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectObjects-response)))
  "Returns full string definition for message of type 'DetectObjects-response"
  (cl:format cl:nil "DetectedObject object~%~%~%================================================================================~%MSG: robot_controller/DetectedObject~%int32 x1~%int32 y1~%int32 x2~%int32 y2~%string class_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectObjects-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'object))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectObjects-response>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectObjects-response
    (cl:cons ':object (object msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'DetectObjects)))
  'DetectObjects-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'DetectObjects)))
  'DetectObjects-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectObjects)))
  "Returns string type for a service object of type '<DetectObjects>"
  "robot_controller/DetectObjects")