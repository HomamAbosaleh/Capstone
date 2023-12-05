
(cl:in-package :asdf)

(defsystem "robot_controller-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :robot_controller-msg
               :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "DetectObjects" :depends-on ("_package_DetectObjects"))
    (:file "_package_DetectObjects" :depends-on ("_package"))
  ))