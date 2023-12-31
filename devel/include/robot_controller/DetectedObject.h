// Generated by gencpp from file robot_controller/DetectedObject.msg
// DO NOT EDIT!


#ifndef ROBOT_CONTROLLER_MESSAGE_DETECTEDOBJECT_H
#define ROBOT_CONTROLLER_MESSAGE_DETECTEDOBJECT_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace robot_controller
{
template <class ContainerAllocator>
struct DetectedObject_
{
  typedef DetectedObject_<ContainerAllocator> Type;

  DetectedObject_()
    : x1(0)
    , y1(0)
    , x2(0)
    , y2(0)
    , class_name()  {
    }
  DetectedObject_(const ContainerAllocator& _alloc)
    : x1(0)
    , y1(0)
    , x2(0)
    , y2(0)
    , class_name(_alloc)  {
  (void)_alloc;
    }



   typedef int32_t _x1_type;
  _x1_type x1;

   typedef int32_t _y1_type;
  _y1_type y1;

   typedef int32_t _x2_type;
  _x2_type x2;

   typedef int32_t _y2_type;
  _y2_type y2;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _class_name_type;
  _class_name_type class_name;





  typedef boost::shared_ptr< ::robot_controller::DetectedObject_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::robot_controller::DetectedObject_<ContainerAllocator> const> ConstPtr;

}; // struct DetectedObject_

typedef ::robot_controller::DetectedObject_<std::allocator<void> > DetectedObject;

typedef boost::shared_ptr< ::robot_controller::DetectedObject > DetectedObjectPtr;
typedef boost::shared_ptr< ::robot_controller::DetectedObject const> DetectedObjectConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::robot_controller::DetectedObject_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::robot_controller::DetectedObject_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::robot_controller::DetectedObject_<ContainerAllocator1> & lhs, const ::robot_controller::DetectedObject_<ContainerAllocator2> & rhs)
{
  return lhs.x1 == rhs.x1 &&
    lhs.y1 == rhs.y1 &&
    lhs.x2 == rhs.x2 &&
    lhs.y2 == rhs.y2 &&
    lhs.class_name == rhs.class_name;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::robot_controller::DetectedObject_<ContainerAllocator1> & lhs, const ::robot_controller::DetectedObject_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace robot_controller

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::robot_controller::DetectedObject_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robot_controller::DetectedObject_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robot_controller::DetectedObject_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robot_controller::DetectedObject_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robot_controller::DetectedObject_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robot_controller::DetectedObject_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::robot_controller::DetectedObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "1bb2b1cebc922acde2c27fad3f257e15";
  }

  static const char* value(const ::robot_controller::DetectedObject_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x1bb2b1cebc922acdULL;
  static const uint64_t static_value2 = 0xe2c27fad3f257e15ULL;
};

template<class ContainerAllocator>
struct DataType< ::robot_controller::DetectedObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "robot_controller/DetectedObject";
  }

  static const char* value(const ::robot_controller::DetectedObject_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::robot_controller::DetectedObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int32 x1\n"
"int32 y1\n"
"int32 x2\n"
"int32 y2\n"
"string class_name\n"
;
  }

  static const char* value(const ::robot_controller::DetectedObject_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::robot_controller::DetectedObject_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.x1);
      stream.next(m.y1);
      stream.next(m.x2);
      stream.next(m.y2);
      stream.next(m.class_name);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DetectedObject_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::robot_controller::DetectedObject_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::robot_controller::DetectedObject_<ContainerAllocator>& v)
  {
    s << indent << "x1: ";
    Printer<int32_t>::stream(s, indent + "  ", v.x1);
    s << indent << "y1: ";
    Printer<int32_t>::stream(s, indent + "  ", v.y1);
    s << indent << "x2: ";
    Printer<int32_t>::stream(s, indent + "  ", v.x2);
    s << indent << "y2: ";
    Printer<int32_t>::stream(s, indent + "  ", v.y2);
    s << indent << "class_name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.class_name);
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROBOT_CONTROLLER_MESSAGE_DETECTEDOBJECT_H
