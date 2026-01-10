project(TrackerEvent)
cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)
aux_source_directory(src MODULE_SRC)

find_package(ROOT REQUIRED COMPONENTS RIO Net)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories ("${ROOT_INCLUDE_DIRS}")

# module includes and ROOT dictionary for tracker_event class
add_library(trk_event_dict SHARED DetectorEvent.h LinkDef.h)
set(TRACKER_EVENT_DICT trk_event_dict)
ROOT_GENERATE_DICTIONARY(tracker_dict DetectorEvent.h MODULE ${TRACKER_EVENT_DICT} LINKDEF LinkDef.h)
target_link_libraries(${TRACKER_EVENT_DICT} ${ROOT_LIBRARIES})

install(TARGETS ${TRACKER_EVENT_DICT}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# copy ROOT dictionary files for runtime access
set(TRACKER_EVENT_PCM ${CMAKE_CURRENT_BINARY_DIR}/libtrk_event_dict_rdict.pcm)
set(TRACKER_EVENT_MAP ${CMAKE_CURRENT_BINARY_DIR}/libtrk_event_dict.rootmap)
install(FILES ${TRACKER_EVENT_PCM} ${TRACKER_EVENT_MAP} DESTINATION lib)
