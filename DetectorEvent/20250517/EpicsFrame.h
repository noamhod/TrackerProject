#ifndef EpicsFrame_h
#define EpicsFrame_h

#include <cstdint>
#include <map>

struct epics_frame {
    struct epics_pv {
	std::string name;
	std::string value;
	std::string ca_server_timestamp;
    };
    std::map<std::string, epics_pv> pv_map;
};

#endif
