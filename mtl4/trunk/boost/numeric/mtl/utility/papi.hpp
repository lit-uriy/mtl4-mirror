// $COPYRIGHT$

#ifndef MTL_PAPI_INCLUDE
#define MTL_PAPI_INCLUDE

#ifdef MTL_HAS_PAPI

#include <papi.h>

#endif // MTL_HAS_PAPI

namespace mtl { namespace utility {


struct papi_error {};
struct papi_version_mismatch : papi_error {};
struct papi_no_counters : papi_error {};
struct papi_create_eventset_error : papi_error {};
struct papi_name_to_code_error : papi_error {};
struct papi_query_event_error : papi_error {};
struct papi_start_event_error : papi_error {};
struct papi_add_event_error : papi_error {};
struct papi_reset_error : papi_error {};
struct papi_read_error : papi_error {};
struct papi_index_range_error : papi_error {};

#ifdef MTL_HAS_PAPI

class papi_t 
{
    void init_papi()
    {
	static bool initialized= false;
	if (!initialized) {

	    if ( PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT )
		throw papi_version_mismatch();

	    num_counters = PAPI_get_opt(PAPI_MAX_HWCTRS, NULL);
	    if (num_counters <= 0) throw papi_no_counters();

	    counters= new long_long[num_counters];

	    if (PAPI_create_eventset(&event_set) != PAPI_OK) throw papi_create_eventset_error();
	    initialized= true;
	}
    }

public:
    const static bool true_papi = true;

    papi_t() : event_set(PAPI_NULL), active_events(0)
    {
	init_papi();
    }


    ~papi_t()
    {
	delete[](counters);
    }


    // returns index of added event
    int add_event(const char* name)
    {
	int code;
	if (PAPI_event_name_to_code(const_cast<char*>(name), &code) != PAPI_OK) 
	    throw papi_name_to_code_error();
	// std::cout << "add event " << const_cast<char*>(name) << " " << code << "\n";
	if (PAPI_query_event(code) != PAPI_OK)
	    throw papi_query_event_error();
	if (PAPI_add_event(event_set, code) != PAPI_OK) 
	    throw papi_add_event_error();
	list_events();
	return active_events++;
    }

    void start() 
    {
	if (PAPI_start(event_set) != PAPI_OK)
	    throw papi_start_event_error();
	reset();
    }

    void list_events()
    {
#if 0
	int evv[8], num= 8;
	PAPI_list_events(event_set, evv, &num);
	for (int i= 0; i < num; i++ ) std::cout << evv[i] << "\n";
#endif
    }

    bool is_event_supported(const char* name) const
    {
	int code;
	return PAPI_event_name_to_code(const_cast<char*>(name), &code) == PAPI_OK 
	       && PAPI_query_event(code) == PAPI_OK;
    }

    void reset()
    {
	if (PAPI_reset(event_set) != PAPI_OK) throw papi_reset_error();
    }

    void read()
    {
	list_events();
	if (PAPI_read(event_set, counters) != PAPI_OK) throw papi_read_error();
	// std::cout << "counters read, first value: " << *counters << "\n";
    }

    long_long operator[](int index) const
    {
	if (index < 0 || index >= active_events) throw papi_index_range_error();
	return counters[index];
    }

    //private:
    int num_counters, event_set, active_events;
    long_long *counters;
};

#else // no papi

// Faked papi type:

struct papi_t
{
    const static bool true_papi = false;
    int add_event(const char* name) { return 0;}
    void start() {}
    bool is_event_supported(const char* name) const { return false;}
    void reset() {}
    void read() {}
    long long operator[](int index) const { return 0; }
};
 
#endif

}} // namespace mtl::utility


#endif // MTL_PAPI_INCLUDE
