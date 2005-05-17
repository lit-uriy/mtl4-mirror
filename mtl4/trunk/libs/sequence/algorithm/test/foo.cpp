#line 1 "copy.cpp"




#line 1 "..\\..\\..\\..\\boost/sequence/algorithm/copy.hpp"






#line 1 "..\\..\\..\\..\\boost/sequence/algorithm/dispatch.hpp"






#line 1 "..\\..\\..\\..\\boost/sequence/category.hpp"






#line 1 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"






#line 1 "..\\..\\..\\..\\boost/sequence/category_fwd.hpp"






namespace boost { namespace sequence { 

template <class Sequence, class Enable = void> struct category;

}} 

#line 14 "..\\..\\..\\..\\boost/sequence/category_fwd.hpp"
#line 8 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"
#line 1 "..\\..\\..\\..\\boost/sequence/fixed_size/is_fixed_size.hpp"






#line 1 "c:\\boost\\boost/array.hpp"


























#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

#pragma once






		


		



		

 
  

 

#line 24 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

 
  
 #line 28 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

  







#line 38 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"















#pragma once
#line 18 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"







#pragma comment(lib,"msvcprtd")


#line 29 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"







#line 37 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"







#line 45 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"

#line 47 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\use_ansi.h"
#line 40 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"


 
#line 44 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"


 

   
    

   

#line 54 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

 #line 56 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

 
  
 #line 60 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"


  



		

 
  
  
  




  
  
  

  







   
   
   
  #line 92 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

  
  
  
  

 












#line 112 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

 

 
namespace std {
typedef bool _Bool;
}
 #line 120 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

		





		






typedef __int64 _Longlong;
typedef unsigned __int64 _ULonglong;

		


 
  
 #line 143 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"





		
		





 
namespace std {
		
class __declspec(dllimport) _Lockit
	{	
public:
  

	explicit _Lockit();	
	explicit _Lockit(int);	
	~_Lockit();	

private:
	_Lockit(const _Lockit&);				
	_Lockit& operator=(const _Lockit&);	

	int _Locktype;

  












#line 187 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

	};

class __declspec(dllimport) _Mutex
	{	
public:

  
	_Mutex();
	~_Mutex();
	void _Lock();
	void _Unlock();

private:
	_Mutex(const _Mutex&);				
	_Mutex& operator=(const _Mutex&);	
	void *_Mtx;

  







#line 214 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

	};

class _Init_locks
	{	
public:

 
	_Init_locks();
	~_Init_locks();

 







#line 234 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"

	};
}
 #line 238 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"



		

extern "C" {
__declspec(dllimport) void __cdecl _Atexit(void (__cdecl *)(void));
}

typedef int _Mbstatet;





#line 254 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\yvals.h"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"







 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"















#pragma once
#line 18 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"






#line 25 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"



extern "C" {
#line 30 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"







#line 38 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"
#line 39 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"








#line 48 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"
#line 49 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"





#line 55 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"









#line 65 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"
#line 66 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"





__declspec(dllimport) extern int * __cdecl _errno(void);



#line 76 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"








typedef __w64 int            intptr_t;
#line 86 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"

#line 88 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"





typedef __w64 unsigned int   uintptr_t;
#line 95 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"

#line 97 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"





typedef __w64 int            ptrdiff_t;
#line 104 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"

#line 106 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"






typedef __w64 unsigned int   size_t;
#line 114 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"

#line 116 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"













#line 130 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"



__declspec(dllimport) extern unsigned long  __cdecl __threadid(void);

__declspec(dllimport) extern uintptr_t __cdecl __threadhandle(void);
#line 137 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"



}
#line 142 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"

#line 144 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stddef.h"
#line 14 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"

 
namespace std {
using ::ptrdiff_t; using ::size_t;
}
 #line 20 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"

#line 22 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"
#line 23 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstddef"





#line 28 "c:\\boost\\boost/array.hpp"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstddef"

#pragma once








#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

		

 
 
 
 
 

 
 

 
 
 

 











#line 43 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstddef"

		
 

 

		




 
 

 

 
  
  
  
 #line 64 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstddef"

		
enum _Uninitialized
	{	
	_Noinit};

		
__declspec(dllimport) void __cdecl _Nomemory();
}
#pragma warning(pop)
#pragma pack(pop)

#line 77 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstddef"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

  




 
 }

 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"













#pragma once
#line 16 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"






#line 23 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"





#pragma pack(push,8)
#line 30 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"















typedef void (__cdecl *terminate_function)();
typedef void (__cdecl *unexpected_function)();
typedef void (__cdecl *terminate_handler)();
typedef void (__cdecl *unexpected_handler)();

struct _EXCEPTION_POINTERS;
typedef void (__cdecl *_se_translator_function)(unsigned int, struct _EXCEPTION_POINTERS*);


__declspec(dllimport) __declspec(noreturn) void __cdecl terminate(void);
__declspec(dllimport) __declspec(noreturn) void __cdecl unexpected(void);



#line 60 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"

__declspec(dllimport) terminate_function __cdecl set_terminate(terminate_function);
__declspec(dllimport) unexpected_function __cdecl set_unexpected(unexpected_function);
__declspec(dllimport) _se_translator_function __cdecl _set_se_translator(_se_translator_function);
__declspec(dllimport) bool __cdecl __uncaught_exception();


#pragma pack(pop)
#line 69 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"

#line 71 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\eh.h"
#line 20 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

 

#line 24 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

 










typedef const char *__exString;

class __declspec(dllimport) exception
	{	
public:
	exception();
	exception(const char *const&);
	exception(const exception&);
	exception& operator=(const exception&);
	virtual ~exception();
	virtual const char *what() const;

private:
	const char *_m_what;
	int _m_doFree;
	};

 namespace std {

using ::exception; using ::set_terminate; using ::terminate_handler; using ::terminate; using ::set_unexpected; using ::unexpected_handler; using ::unexpected;
typedef void (*_Prhand)(const exception&);
extern __declspec(dllimport) _Prhand _Raise_handler;
__declspec(dllimport) bool __cdecl uncaught_exception();

 
























































































#line 150 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

		
class bad_exception : public exception
	{	
public:
	bad_exception(const char *_Message = "bad exception")
		throw ()
		: exception(_Message)
		{	
		}

	virtual ~bad_exception() throw ()
		{	
		}

 





#line 172 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"

	};
}
#pragma warning(pop)
#pragma pack(pop)

#line 179 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\exception"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstring"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdlib"

#pragma once










 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
















#pragma once
#line 19 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"






#line 26 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"







#pragma pack(push,8)
#line 35 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


extern "C" {
#line 39 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"








#line 48 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
















#line 65 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"




































typedef int (__cdecl * _onexit_t)(void);



#line 106 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

#line 108 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"






typedef struct _div_t {
        int quot;
        int rem;
} div_t;

typedef struct _ldiv_t {
        long quot;
        long rem;
} ldiv_t;


#line 126 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"












__declspec(dllimport) extern int __mb_cur_max;

__declspec(dllimport) int __cdecl ___mb_cur_max_func(void);
#line 142 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"






























typedef void (__cdecl * _secerr_handler_func)(int, void *);
#line 174 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


typedef void (__cdecl *_purecall_handler)(); 

__declspec(dllimport) _purecall_handler __cdecl _set_purecall_handler(_purecall_handler);




__declspec(dllimport) int * __cdecl _errno(void);
__declspec(dllimport) unsigned long * __cdecl __doserrno(void);





#line 191 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


__declspec(dllimport) extern char * _sys_errlist[];   
__declspec(dllimport) extern int _sys_nerr;           












#line 208 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"



__declspec(dllimport) int *          __cdecl __p___argc(void);
__declspec(dllimport) char ***       __cdecl __p___argv(void);
__declspec(dllimport) wchar_t ***    __cdecl __p___wargv(void);
__declspec(dllimport) char ***       __cdecl __p__environ(void);
__declspec(dllimport) wchar_t ***    __cdecl __p__wenviron(void);
__declspec(dllimport) char **        __cdecl __p__pgmptr(void);
__declspec(dllimport) wchar_t **     __cdecl __p__wpgmptr(void);


















#line 237 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


__declspec(dllimport) extern int _fmode;          
__declspec(dllimport) extern int _fileinfo;       




__declspec(dllimport) extern unsigned int _osplatform;
__declspec(dllimport) extern unsigned int _osver;
__declspec(dllimport) extern unsigned int _winver;
__declspec(dllimport) extern unsigned int _winmajor;
__declspec(dllimport) extern unsigned int _winminor;





__declspec(dllimport) __declspec(noreturn) void   __cdecl abort(void);
__declspec(dllimport) __declspec(noreturn) void   __cdecl exit(int);



#line 261 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"



#line 265 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
        int    __cdecl abs(int);
#line 267 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
        __int64    __cdecl _abs64(__int64);
        int    __cdecl atexit(void (__cdecl *)(void));
__declspec(dllimport) double __cdecl atof(const char *);
__declspec(dllimport) int    __cdecl atoi(const char *);
__declspec(dllimport) long   __cdecl atol(const char *);
__declspec(dllimport) void * __cdecl bsearch(const void *, const void *, size_t, size_t,
        int (__cdecl *)(const void *, const void *));
        unsigned short __cdecl _byteswap_ushort(unsigned short);
        unsigned long  __cdecl _byteswap_ulong (unsigned long);
        unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64);
__declspec(dllimport) void * __cdecl calloc(size_t, size_t);
__declspec(dllimport) div_t  __cdecl div(int, int);
__declspec(dllimport) void   __cdecl free(void *);
__declspec(dllimport) char * __cdecl getenv(const char *);
__declspec(dllimport) char * __cdecl _itoa(int, char *, int);

__declspec(dllimport) char * __cdecl _i64toa(__int64, char *, int);
__declspec(dllimport) char * __cdecl _ui64toa(unsigned __int64, char *, int);
__declspec(dllimport) __int64 __cdecl _atoi64(const char *);
__declspec(dllimport) __int64 __cdecl _strtoi64(const char *, char **, int);
__declspec(dllimport) unsigned __int64 __cdecl _strtoui64(const char *, char **, int);
#line 289 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


#line 292 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
        long __cdecl labs(long);
#line 294 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
__declspec(dllimport) ldiv_t __cdecl ldiv(long, long);
__declspec(dllimport) char * __cdecl _ltoa(long, char *, int);
__declspec(dllimport) void * __cdecl malloc(size_t);
__declspec(dllimport) int    __cdecl mblen(const char *, size_t);
__declspec(dllimport) size_t __cdecl _mbstrlen(const char *s);
__declspec(dllimport) int    __cdecl mbtowc(wchar_t *, const char *, size_t);
__declspec(dllimport) size_t __cdecl mbstowcs(wchar_t *, const char *, size_t);
__declspec(dllimport) void   __cdecl qsort(void *, size_t, size_t, int (__cdecl *)
        (const void *, const void *));
__declspec(dllimport) int    __cdecl rand(void);
__declspec(dllimport) void * __cdecl realloc(void *, size_t);
__declspec(dllimport) int    __cdecl _set_error_mode(int);

__declspec(dllimport) _secerr_handler_func
               __cdecl _set_security_error_handler(_secerr_handler_func);
#line 310 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
__declspec(dllimport) void   __cdecl srand(unsigned int);
__declspec(dllimport) double __cdecl strtod(const char *, char **);
__declspec(dllimport) long   __cdecl strtol(const char *, char **, int);
__declspec(dllimport) unsigned long __cdecl strtoul(const char *, char **, int);
__declspec(dllimport) int    __cdecl system(const char *);
__declspec(dllimport) char * __cdecl _ultoa(unsigned long, char *, int);
__declspec(dllimport) int    __cdecl wctomb(char *, wchar_t);
__declspec(dllimport) size_t __cdecl wcstombs(char *, const wchar_t *, size_t);






__declspec(dllimport) wchar_t * __cdecl _itow (int, wchar_t *, int);
__declspec(dllimport) wchar_t * __cdecl _ltow (long, wchar_t *, int);
__declspec(dllimport) wchar_t * __cdecl _ultow (unsigned long, wchar_t *, int);
__declspec(dllimport) double __cdecl wcstod(const wchar_t *, wchar_t **);
__declspec(dllimport) long   __cdecl wcstol(const wchar_t *, wchar_t **, int);
__declspec(dllimport) unsigned long __cdecl wcstoul(const wchar_t *, wchar_t **, int);
__declspec(dllimport) wchar_t * __cdecl _wgetenv(const wchar_t *);
__declspec(dllimport) int    __cdecl _wsystem(const wchar_t *);
__declspec(dllimport) double __cdecl _wtof(const wchar_t *);
__declspec(dllimport) int __cdecl _wtoi(const wchar_t *);
__declspec(dllimport) long __cdecl _wtol(const wchar_t *);

__declspec(dllimport) wchar_t * __cdecl _i64tow(__int64, wchar_t *, int);
__declspec(dllimport) wchar_t * __cdecl _ui64tow(unsigned __int64, wchar_t *, int);
__declspec(dllimport) __int64   __cdecl _wtoi64(const wchar_t *);
__declspec(dllimport) __int64   __cdecl _wcstoi64(const wchar_t *, wchar_t **, int);
__declspec(dllimport) unsigned __int64  __cdecl _wcstoui64(const wchar_t *, wchar_t **, int);
#line 342 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


#line 345 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"









__declspec(dllimport) char * __cdecl _ecvt(double, int, int *, int *);

__declspec(dllimport) __declspec(noreturn) void   __cdecl _exit(int);


#line 360 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
__declspec(dllimport) char * __cdecl _fcvt(double, int, int *, int *);
__declspec(dllimport) char * __cdecl _fullpath(char *, const char *, size_t);
__declspec(dllimport) char * __cdecl _gcvt(double, int, char *);
        unsigned long __cdecl _lrotl(unsigned long, int);
        unsigned long __cdecl _lrotr(unsigned long, int);
__declspec(dllimport) void   __cdecl _makepath(char *, const char *, const char *, const char *,
        const char *);
        _onexit_t __cdecl _onexit(_onexit_t);
__declspec(dllimport) void   __cdecl perror(const char *);
__declspec(dllimport) int    __cdecl _putenv(const char *);
        unsigned int __cdecl _rotl(unsigned int, int);
        unsigned __int64 __cdecl _rotl64(unsigned __int64, int);
        unsigned int __cdecl _rotr(unsigned int, int);
        unsigned __int64 __cdecl _rotr64(unsigned __int64, int);
__declspec(dllimport) void   __cdecl _searchenv(const char *, const char *, char *);
__declspec(dllimport) void   __cdecl _splitpath(const char *, char *, char *, char *, char *);
__declspec(dllimport) void   __cdecl _swab(char *, char *, int);





__declspec(dllimport) wchar_t * __cdecl _wfullpath(wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) void   __cdecl _wmakepath(wchar_t *, const wchar_t *, const wchar_t *, const wchar_t *,
        const wchar_t *);
__declspec(dllimport) void   __cdecl _wperror(const wchar_t *);
__declspec(dllimport) int    __cdecl _wputenv(const wchar_t *);
__declspec(dllimport) void   __cdecl _wsearchenv(const wchar_t *, const wchar_t *, wchar_t *);
__declspec(dllimport) void   __cdecl _wsplitpath(const wchar_t *, wchar_t *, wchar_t *, wchar_t *, wchar_t *);


#line 392 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"



__declspec(dllimport) void __cdecl _seterrormode(int);
__declspec(dllimport) void __cdecl _beep(unsigned, unsigned);
__declspec(dllimport) void __cdecl _sleep(unsigned long);


#line 401 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"







__declspec(dllimport) int __cdecl tolower(int);
#line 410 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

__declspec(dllimport) int __cdecl toupper(int);
#line 413 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

#line 415 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

















__declspec(dllimport) char * __cdecl ecvt(double, int, int *, int *);
__declspec(dllimport) char * __cdecl fcvt(double, int, int *, int *);
__declspec(dllimport) char * __cdecl gcvt(double, int, char *);
__declspec(dllimport) char * __cdecl itoa(int, char *, int);
__declspec(dllimport) char * __cdecl ltoa(long, char *, int);
        _onexit_t __cdecl onexit(_onexit_t);
__declspec(dllimport) int    __cdecl putenv(const char *);
__declspec(dllimport) void   __cdecl swab(char *, char *, int);
__declspec(dllimport) char * __cdecl ultoa(unsigned long, char *, int);

#line 443 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

#line 445 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


}

#line 450 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"


#pragma pack(pop)
#line 454 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"

#line 456 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdlib.h"
#line 14 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdlib"

 
namespace std {
using ::size_t; using ::div_t; using ::ldiv_t;

using ::abort; using ::abs; using ::atexit;
using ::atof; using ::atoi; using ::atol;
using ::bsearch; using ::calloc; using ::div;
using ::exit; using ::free; using ::getenv;
using ::labs; using ::ldiv; using ::malloc;
using ::mblen; using ::mbstowcs; using ::mbtowc;
using ::qsort; using ::rand; using ::realloc;
using ::srand; using ::strtod; using ::strtol;
using ::strtoul; using ::system;
using ::wcstombs; using ::wctomb;
}
 #line 31 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdlib"

#line 33 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdlib"
#line 34 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdlib"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

#pragma once




#pragma pack(push,8)
#pragma warning(push,3)

  
  

namespace std {
		
class bad_alloc
	: public exception
	{	
public:
	bad_alloc(const char *_Message = "bad allocation") throw ()
		: exception(_Message)
		{	
		}

	virtual ~bad_alloc() throw ()
		{	
		}

 





#line 35 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

	};

		
 
typedef void (__cdecl *new_handler)();	
 #line 42 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

 
struct nothrow_t
	{	
	};

extern const nothrow_t nothrow;	
 #line 50 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

		
__declspec(dllimport) new_handler __cdecl set_new_handler(new_handler)
	throw ();	
}

		
void __cdecl operator delete(void *) throw ();
void *__cdecl operator new(size_t) throw (...);

 
  
inline void *__cdecl operator new(size_t, void *_Where) throw ()
	{	
	return (_Where);
	}

inline void __cdecl operator delete(void *, void *) throw ()
	{	
	}
 #line 71 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

 
  
inline void *__cdecl operator new[](size_t, void *_Where) throw ()
	{	
	return (_Where);
	}

inline void __cdecl operator delete[](void *, void *) throw ()
	{	
	}
 #line 83 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

void __cdecl operator delete[](void *) throw ();	

void *__cdecl operator new[](size_t)
	throw (...);	

 
  
void *__cdecl operator new(size_t, const std::nothrow_t&)
	throw ();

void *__cdecl operator new[](size_t, const std::nothrow_t&)
	throw ();	

void __cdecl operator delete(void *, const std::nothrow_t&)
	throw ();	

void __cdecl operator delete[](void *, const std::nothrow_t&)
	throw ();	
 #line 103 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"



 
using std::new_handler;
 #line 109 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"

  

#pragma warning(pop)
#pragma pack(pop)

#line 116 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\new"





#line 7 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\climits"

#pragma once




 #pragma warning(disable: 4514)

#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"















#pragma once
#line 18 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"






#line 25 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"













#line 39 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"

















#line 57 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"





#line 63 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"





#line 69 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"





#line 75 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"








#line 84 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"








#line 93 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"
































#line 126 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\limits.h"
#line 10 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\climits"
#line 11 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\climits"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\utility"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdio"

#pragma once










 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"















#pragma once
#line 18 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"






#line 25 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"







#pragma pack(push,8)
#line 34 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


extern "C" {
#line 38 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"








#line 47 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"
















#line 64 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"



















typedef unsigned short wint_t;
typedef unsigned short wctype_t;

#line 87 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"









typedef char *  va_list;
#line 98 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

#line 100 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

























struct _iobuf {
        char *_ptr;
        int   _cnt;
        char *_base;
        int   _flag;
        int   _file;
        int   _charbuf;
        int   _bufsiz;
        char *_tmpfname;
        };
typedef struct _iobuf FILE;

#line 138 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"










#line 149 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"










































__declspec(dllimport) extern FILE _iob[];
#line 193 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"










#line 204 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


typedef __int64 fpos_t;







#line 215 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"
#line 216 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


#line 219 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"




























__declspec(dllimport) int __cdecl _filbuf(FILE *);
__declspec(dllimport) int __cdecl _flsbuf(int, FILE *);




__declspec(dllimport) FILE * __cdecl _fsopen(const char *, const char *, int);
#line 255 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

__declspec(dllimport) void __cdecl clearerr(FILE *);
__declspec(dllimport) int __cdecl fclose(FILE *);
__declspec(dllimport) int __cdecl _fcloseall(void);




__declspec(dllimport) FILE * __cdecl _fdopen(int, const char *);
#line 265 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

__declspec(dllimport) int __cdecl feof(FILE *);
__declspec(dllimport) int __cdecl ferror(FILE *);
__declspec(dllimport) int __cdecl fflush(FILE *);
__declspec(dllimport) int __cdecl fgetc(FILE *);
__declspec(dllimport) int __cdecl _fgetchar(void);
__declspec(dllimport) int __cdecl fgetpos(FILE *, fpos_t *);
__declspec(dllimport) char * __cdecl fgets(char *, int, FILE *);




__declspec(dllimport) int __cdecl _fileno(FILE *);
#line 279 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

__declspec(dllimport) int __cdecl _flushall(void);
__declspec(dllimport) FILE * __cdecl fopen(const char *, const char *);
__declspec(dllimport) int __cdecl fprintf(FILE *, const char *, ...);
__declspec(dllimport) int __cdecl fputc(int, FILE *);
__declspec(dllimport) int __cdecl _fputchar(int);
__declspec(dllimport) int __cdecl fputs(const char *, FILE *);
__declspec(dllimport) size_t __cdecl fread(void *, size_t, size_t, FILE *);
__declspec(dllimport) FILE * __cdecl freopen(const char *, const char *, FILE *);
__declspec(dllimport) int __cdecl fscanf(FILE *, const char *, ...);
__declspec(dllimport) int __cdecl fsetpos(FILE *, const fpos_t *);
__declspec(dllimport) int __cdecl fseek(FILE *, long, int);
__declspec(dllimport) long __cdecl ftell(FILE *);
__declspec(dllimport) size_t __cdecl fwrite(const void *, size_t, size_t, FILE *);
__declspec(dllimport) int __cdecl getc(FILE *);
__declspec(dllimport) int __cdecl getchar(void);
__declspec(dllimport) int __cdecl _getmaxstdio(void);
__declspec(dllimport) char * __cdecl gets(char *);
__declspec(dllimport) int __cdecl _getw(FILE *);
__declspec(dllimport) void __cdecl perror(const char *);
__declspec(dllimport) int __cdecl _pclose(FILE *);
__declspec(dllimport) FILE * __cdecl _popen(const char *, const char *);
__declspec(dllimport) int __cdecl printf(const char *, ...);
__declspec(dllimport) int __cdecl putc(int, FILE *);
__declspec(dllimport) int __cdecl putchar(int);
__declspec(dllimport) int __cdecl puts(const char *);
__declspec(dllimport) int __cdecl _putw(int, FILE *);
__declspec(dllimport) int __cdecl remove(const char *);
__declspec(dllimport) int __cdecl rename(const char *, const char *);
__declspec(dllimport) void __cdecl rewind(FILE *);
__declspec(dllimport) int __cdecl _rmtmp(void);
__declspec(dllimport) int __cdecl scanf(const char *, ...);
__declspec(dllimport) void __cdecl setbuf(FILE *, char *);
__declspec(dllimport) int __cdecl _setmaxstdio(int);
__declspec(dllimport) int __cdecl setvbuf(FILE *, char *, int, size_t);
__declspec(dllimport) int __cdecl _snprintf(char *, size_t, const char *, ...);
__declspec(dllimport) int __cdecl sprintf(char *, const char *, ...);
__declspec(dllimport) int __cdecl _scprintf(const char *, ...);
__declspec(dllimport) int __cdecl sscanf(const char *, const char *, ...);
__declspec(dllimport) int __cdecl _snscanf(const char *, size_t, const char *, ...);
__declspec(dllimport) char * __cdecl _tempnam(const char *, const char *);
__declspec(dllimport) FILE * __cdecl tmpfile(void);
__declspec(dllimport) char * __cdecl tmpnam(char *);
__declspec(dllimport) int __cdecl ungetc(int, FILE *);
__declspec(dllimport) int __cdecl _unlink(const char *);
__declspec(dllimport) int __cdecl vfprintf(FILE *, const char *, va_list);
__declspec(dllimport) int __cdecl vprintf(const char *, va_list);
__declspec(dllimport) int __cdecl _vsnprintf(char *, size_t, const char *, va_list);
__declspec(dllimport) int __cdecl vsprintf(char *, const char *, va_list);
__declspec(dllimport) int __cdecl _vscprintf(const char *, va_list);







#line 337 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"




__declspec(dllimport) FILE * __cdecl _wfsopen(const wchar_t *, const wchar_t *, int);
#line 343 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

__declspec(dllimport) wint_t __cdecl fgetwc(FILE *);
__declspec(dllimport) wint_t __cdecl _fgetwchar(void);
__declspec(dllimport) wint_t __cdecl fputwc(wchar_t, FILE *);
__declspec(dllimport) wint_t __cdecl _fputwchar(wchar_t);
__declspec(dllimport) wint_t __cdecl getwc(FILE *);
__declspec(dllimport) wint_t __cdecl getwchar(void);
__declspec(dllimport) wint_t __cdecl putwc(wchar_t, FILE *);
__declspec(dllimport) wint_t __cdecl putwchar(wchar_t);
__declspec(dllimport) wint_t __cdecl ungetwc(wint_t, FILE *);

__declspec(dllimport) wchar_t * __cdecl fgetws(wchar_t *, int, FILE *);
__declspec(dllimport) int __cdecl fputws(const wchar_t *, FILE *);
__declspec(dllimport) wchar_t * __cdecl _getws(wchar_t *);
__declspec(dllimport) int __cdecl _putws(const wchar_t *);

__declspec(dllimport) int __cdecl fwprintf(FILE *, const wchar_t *, ...);
__declspec(dllimport) int __cdecl wprintf(const wchar_t *, ...);
__declspec(dllimport) int __cdecl _snwprintf(wchar_t *, size_t, const wchar_t *, ...);

__declspec(dllimport) int __cdecl swprintf(wchar_t *, const wchar_t *, ...);


extern "C++" __declspec(dllimport) int __cdecl swprintf(wchar_t *, size_t, const wchar_t *, ...);
#line 368 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"
__declspec(dllimport) int __cdecl _scwprintf(const wchar_t *, ...);
__declspec(dllimport) int __cdecl vfwprintf(FILE *, const wchar_t *, va_list);
__declspec(dllimport) int __cdecl vwprintf(const wchar_t *, va_list);
__declspec(dllimport) int __cdecl _vsnwprintf(wchar_t *, size_t, const wchar_t *, va_list);

__declspec(dllimport) int __cdecl vswprintf(wchar_t *, const wchar_t *, va_list);


extern "C++" __declspec(dllimport) int __cdecl vswprintf(wchar_t *, size_t, const wchar_t *, va_list);
#line 378 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"
__declspec(dllimport) int __cdecl _vscwprintf(const wchar_t *, va_list);
__declspec(dllimport) int __cdecl fwscanf(FILE *, const wchar_t *, ...);
__declspec(dllimport) int __cdecl swscanf(const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) int __cdecl _snwscanf(const wchar_t *, size_t, const wchar_t *, ...);
__declspec(dllimport) int __cdecl wscanf(const wchar_t *, ...);






__declspec(dllimport) FILE * __cdecl _wfdopen(int, const wchar_t *);
__declspec(dllimport) FILE * __cdecl _wfopen(const wchar_t *, const wchar_t *);
__declspec(dllimport) FILE * __cdecl _wfreopen(const wchar_t *, const wchar_t *, FILE *);
__declspec(dllimport) void __cdecl _wperror(const wchar_t *);
__declspec(dllimport) FILE * __cdecl _wpopen(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl _wremove(const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wtempnam(const wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wtmpnam(wchar_t *);



#line 401 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


#line 404 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"





















#line 426 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"










__declspec(dllimport) int __cdecl fcloseall(void);
__declspec(dllimport) FILE * __cdecl fdopen(int, const char *);
__declspec(dllimport) int __cdecl fgetchar(void);
__declspec(dllimport) int __cdecl fileno(FILE *);
__declspec(dllimport) int __cdecl flushall(void);
__declspec(dllimport) int __cdecl fputchar(int);
__declspec(dllimport) int __cdecl getw(FILE *);
__declspec(dllimport) int __cdecl putw(int, FILE *);
__declspec(dllimport) int __cdecl rmtmp(void);
__declspec(dllimport) char * __cdecl tempnam(const char *, const char *);
__declspec(dllimport) int __cdecl unlink(const char *);

#line 449 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


}
#line 453 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"


#pragma pack(pop)
#line 457 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"

#line 459 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdio.h"
#line 14 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdio"

 
namespace std {
using ::size_t; using ::fpos_t; using ::FILE;
using ::clearerr; using ::fclose; using ::feof;
using ::ferror; using ::fflush; using ::fgetc;
using ::fgetpos; using ::fgets; using ::fopen;
using ::fprintf; using ::fputc; using ::fputs;
using ::fread; using ::freopen; using ::fscanf;
using ::fseek; using ::fsetpos; using ::ftell;
using ::fwrite; using ::getc; using ::getchar;
using ::gets; using ::perror;
using ::putc; using ::putchar;
using ::printf; using ::puts; using ::remove;
using ::rename; using ::rewind; using ::scanf;
using ::setbuf; using ::setvbuf; using ::sprintf;
using ::sscanf; using ::tmpfile; using ::tmpnam;
using ::ungetc; using ::vfprintf; using ::vprintf;
using ::vsprintf;
}
 #line 35 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdio"

#line 37 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdio"
#line 38 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstdio"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstring"

#pragma once










 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"















#pragma once
#line 18 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"






#line 25 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"



extern "C" {
#line 30 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"








#line 39 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"















#line 55 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"




















#line 76 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"
























        void *  __cdecl memcpy(void *, const void *, size_t);
        int     __cdecl memcmp(const void *, const void *, size_t);
        void *  __cdecl memset(void *, int, size_t);
        char *  __cdecl _strset(char *, int);
        char *  __cdecl strcpy(char *, const char *);
        char *  __cdecl strcat(char *, const char *);
        int     __cdecl strcmp(const char *, const char *);
        size_t  __cdecl strlen(const char *);
#line 109 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"
__declspec(dllimport) void *  __cdecl _memccpy(void *, const void *, int, size_t);
__declspec(dllimport) void *  __cdecl memchr(const void *, int, size_t);
__declspec(dllimport) int     __cdecl _memicmp(const void *, const void *, size_t);



#line 116 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"
__declspec(dllimport) void *  __cdecl memmove(void *, const void *, size_t);
#line 118 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"


__declspec(dllimport) char *  __cdecl strchr(const char *, int);
__declspec(dllimport) int     __cdecl _strcmpi(const char *, const char *);
__declspec(dllimport) int     __cdecl _stricmp(const char *, const char *);
__declspec(dllimport) int     __cdecl strcoll(const char *, const char *);
__declspec(dllimport) int     __cdecl _stricoll(const char *, const char *);
__declspec(dllimport) int     __cdecl _strncoll(const char *, const char *, size_t);
__declspec(dllimport) int     __cdecl _strnicoll(const char *, const char *, size_t);
__declspec(dllimport) size_t  __cdecl strcspn(const char *, const char *);
__declspec(dllimport) char *  __cdecl _strdup(const char *);
__declspec(dllimport) char *  __cdecl _strerror(const char *);
__declspec(dllimport) char *  __cdecl strerror(int);
__declspec(dllimport) char *  __cdecl _strlwr(char *);
__declspec(dllimport) char *  __cdecl strncat(char *, const char *, size_t);
__declspec(dllimport) int     __cdecl strncmp(const char *, const char *, size_t);
__declspec(dllimport) int     __cdecl _strnicmp(const char *, const char *, size_t);
__declspec(dllimport) char *  __cdecl strncpy(char *, const char *, size_t);
__declspec(dllimport) char *  __cdecl _strnset(char *, int, size_t);
__declspec(dllimport) char *  __cdecl strpbrk(const char *, const char *);
__declspec(dllimport) char *  __cdecl strrchr(const char *, int);
__declspec(dllimport) char *  __cdecl _strrev(char *);
__declspec(dllimport) size_t  __cdecl strspn(const char *, const char *);
__declspec(dllimport) char *  __cdecl strstr(const char *, const char *);
__declspec(dllimport) char *  __cdecl strtok(char *, const char *);
__declspec(dllimport) char *  __cdecl _strupr(char *);
__declspec(dllimport) size_t  __cdecl strxfrm (char *, const char *, size_t);





__declspec(dllimport) void * __cdecl memccpy(void *, const void *, int, size_t);
__declspec(dllimport) int __cdecl memicmp(const void *, const void *, size_t);
__declspec(dllimport) int __cdecl strcmpi(const char *, const char *);
__declspec(dllimport) int __cdecl stricmp(const char *, const char *);
__declspec(dllimport) char * __cdecl strdup(const char *);
__declspec(dllimport) char * __cdecl strlwr(char *);
__declspec(dllimport) int __cdecl strnicmp(const char *, const char *, size_t);
__declspec(dllimport) char * __cdecl strnset(char *, int, size_t);
__declspec(dllimport) char * __cdecl strrev(char *);
        char * __cdecl strset(char *, int);
__declspec(dllimport) char * __cdecl strupr(char *);

#line 163 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"






__declspec(dllimport) wchar_t * __cdecl wcscat(wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcschr(const wchar_t *, wchar_t);
__declspec(dllimport) int __cdecl wcscmp(const wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcscpy(wchar_t *, const wchar_t *);
__declspec(dllimport) size_t __cdecl wcscspn(const wchar_t *, const wchar_t *);
__declspec(dllimport) size_t __cdecl wcslen(const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcsncat(wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) int __cdecl wcsncmp(const wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) wchar_t * __cdecl wcsncpy(wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) wchar_t * __cdecl wcspbrk(const wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcsrchr(const wchar_t *, wchar_t);
__declspec(dllimport) size_t __cdecl wcsspn(const wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcsstr(const wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcstok(wchar_t *, const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wcserror(int);
__declspec(dllimport) wchar_t * __cdecl __wcserror(const wchar_t *);

__declspec(dllimport) wchar_t * __cdecl _wcsdup(const wchar_t *);
__declspec(dllimport) int __cdecl _wcsicmp(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl _wcsnicmp(const wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) wchar_t * __cdecl _wcsnset(wchar_t *, wchar_t, size_t);
__declspec(dllimport) wchar_t * __cdecl _wcsrev(wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wcsset(wchar_t *, wchar_t);

__declspec(dllimport) wchar_t * __cdecl _wcslwr(wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wcsupr(wchar_t *);
__declspec(dllimport) size_t __cdecl wcsxfrm(wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) int __cdecl wcscoll(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl _wcsicoll(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl _wcsncoll(const wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) int __cdecl _wcsnicoll(const wchar_t *, const wchar_t *, size_t);







__declspec(dllimport) wchar_t * __cdecl wcsdup(const wchar_t *);
__declspec(dllimport) int __cdecl wcsicmp(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl wcsnicmp(const wchar_t *, const wchar_t *, size_t);
__declspec(dllimport) wchar_t * __cdecl wcsnset(wchar_t *, wchar_t, size_t);
__declspec(dllimport) wchar_t * __cdecl wcsrev(wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcsset(wchar_t *, wchar_t);
__declspec(dllimport) wchar_t * __cdecl wcslwr(wchar_t *);
__declspec(dllimport) wchar_t * __cdecl wcsupr(wchar_t *);
__declspec(dllimport) int __cdecl wcsicoll(const wchar_t *, const wchar_t *);

#line 218 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"


#line 221 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"



}
#line 226 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"

#line 228 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\string.h"
#line 14 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstring"

 
namespace std {
using ::size_t; using ::memchr; using ::memcmp;
using ::memcpy; using ::memmove; using ::memset;
using ::strcat; using ::strchr; using ::strcmp;
using ::strcoll; using ::strcpy; using ::strcspn;
using ::strerror; using ::strlen; using ::strncat;
using ::strncmp; using ::strncpy; using ::strpbrk;
using ::strrchr; using ::strspn; using ::strstr;
using ::strtok; using ::strxfrm;
}
 #line 27 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstring"

#line 29 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstring"
#line 30 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cstring"





#line 7 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cwchar"

#pragma once










 #line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


















#pragma once
#line 21 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"









#line 31 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



#pragma pack(push,8)
#line 36 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


extern "C" {
#line 40 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"








#line 49 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
























#line 74 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"















typedef __w64 long time_t;       
#line 91 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 93 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



typedef __int64 __time64_t;     
#line 98 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 100 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"































































typedef unsigned long _fsize_t; 

#line 166 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



struct _wfinddata_t {
        unsigned attrib;
        time_t   time_create;   
        time_t   time_access;   
        time_t   time_write;
        _fsize_t size;
        wchar_t  name[260];
};



struct _wfinddatai64_t {
        unsigned attrib;
        time_t   time_create;   
        time_t   time_access;   
        time_t   time_write;
        __int64  size;
        wchar_t  name[260];
};

struct __wfinddata64_t {
        unsigned attrib;
        __time64_t  time_create;    
        __time64_t  time_access;    
        __time64_t  time_write;
        __int64     size;
        wchar_t     name[260];
};

#line 199 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#line 202 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"













__declspec(dllimport) extern const unsigned short _wctype[];
__declspec(dllimport) extern const unsigned short *_pctype;
__declspec(dllimport) extern const wctype_t *_pwctype;
#line 219 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"








                                
















__declspec(dllimport) int __cdecl iswalpha(wint_t);
__declspec(dllimport) int __cdecl iswupper(wint_t);
__declspec(dllimport) int __cdecl iswlower(wint_t);
__declspec(dllimport) int __cdecl iswdigit(wint_t);
__declspec(dllimport) int __cdecl iswxdigit(wint_t);
__declspec(dllimport) int __cdecl iswspace(wint_t);
__declspec(dllimport) int __cdecl iswpunct(wint_t);
__declspec(dllimport) int __cdecl iswalnum(wint_t);
__declspec(dllimport) int __cdecl iswprint(wint_t);
__declspec(dllimport) int __cdecl iswgraph(wint_t);
__declspec(dllimport) int __cdecl iswcntrl(wint_t);
__declspec(dllimport) int __cdecl iswascii(wint_t);
__declspec(dllimport) int __cdecl isleadbyte(int);

__declspec(dllimport) wchar_t __cdecl towupper(wchar_t);
__declspec(dllimport) wchar_t __cdecl towlower(wchar_t);

__declspec(dllimport) int __cdecl iswctype(wint_t, wctype_t);


__declspec(dllimport) int __cdecl is_wctype(wint_t, wctype_t);



#line 269 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





__declspec(dllimport) int __cdecl _wchdir(const wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wgetcwd(wchar_t *, int);
__declspec(dllimport) wchar_t * __cdecl _wgetdcwd(int, wchar_t *, int);
__declspec(dllimport) int __cdecl _wmkdir(const wchar_t *);
__declspec(dllimport) int __cdecl _wrmdir(const wchar_t *);


#line 282 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





__declspec(dllimport) int __cdecl _waccess(const wchar_t *, int);
__declspec(dllimport) int __cdecl _wchmod(const wchar_t *, int);
__declspec(dllimport) int __cdecl _wcreat(const wchar_t *, int);
__declspec(dllimport) intptr_t __cdecl _wfindfirst(wchar_t *, struct _wfinddata_t *);
__declspec(dllimport) int __cdecl _wfindnext(intptr_t, struct _wfinddata_t *);
__declspec(dllimport) int __cdecl _wunlink(const wchar_t *);
__declspec(dllimport) int __cdecl _wrename(const wchar_t *, const wchar_t *);
__declspec(dllimport) int __cdecl _wopen(const wchar_t *, int, ...);
__declspec(dllimport) int __cdecl _wsopen(const wchar_t *, int, int, ...);
__declspec(dllimport) wchar_t * __cdecl _wmktemp(wchar_t *);


__declspec(dllimport) intptr_t __cdecl _wfindfirsti64(wchar_t *, struct _wfinddatai64_t *);
__declspec(dllimport) intptr_t __cdecl _wfindfirst64(wchar_t *, struct __wfinddata64_t *);
__declspec(dllimport) int __cdecl _wfindnexti64(intptr_t, struct _wfinddatai64_t *);
__declspec(dllimport) int __cdecl _wfindnext64(intptr_t, struct __wfinddata64_t *);
#line 304 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#line 307 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





__declspec(dllimport) wchar_t * __cdecl _wsetlocale(int, const wchar_t *);


#line 316 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





__declspec(dllimport) intptr_t __cdecl _wexecl(const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wexecle(const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wexeclp(const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wexeclpe(const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wexecv(const wchar_t *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wexecve(const wchar_t *, const wchar_t * const *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wexecvp(const wchar_t *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wexecvpe(const wchar_t *, const wchar_t * const *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wspawnl(int, const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wspawnle(int, const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wspawnlp(int, const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wspawnlpe(int, const wchar_t *, const wchar_t *, ...);
__declspec(dllimport) intptr_t __cdecl _wspawnv(int, const wchar_t *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wspawnve(int, const wchar_t *, const wchar_t * const *,
        const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wspawnvp(int, const wchar_t *, const wchar_t * const *);
__declspec(dllimport) intptr_t __cdecl _wspawnvpe(int, const wchar_t *, const wchar_t * const *,
        const wchar_t * const *);
__declspec(dllimport) int __cdecl _wsystem(const wchar_t *);


#line 343 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





















inline int __cdecl iswalpha(wint_t _C) {return (iswctype(_C,(0x0100|0x1|0x2))); }
inline int __cdecl iswupper(wint_t _C) {return (iswctype(_C,0x1)); }
inline int __cdecl iswlower(wint_t _C) {return (iswctype(_C,0x2)); }
inline int __cdecl iswdigit(wint_t _C) {return (iswctype(_C,0x4)); }
inline int __cdecl iswxdigit(wint_t _C) {return (iswctype(_C,0x80)); }
inline int __cdecl iswspace(wint_t _C) {return (iswctype(_C,0x8)); }
inline int __cdecl iswpunct(wint_t _C) {return (iswctype(_C,0x10)); }
inline int __cdecl iswalnum(wint_t _C) {return (iswctype(_C,(0x0100|0x1|0x2)|0x4)); }
inline int __cdecl iswprint(wint_t _C)
        {return (iswctype(_C,0x40|0x10|(0x0100|0x1|0x2)|0x4)); }
inline int __cdecl iswgraph(wint_t _C)
        {return (iswctype(_C,0x10|(0x0100|0x1|0x2)|0x4)); }
inline int __cdecl iswcntrl(wint_t _C) {return (iswctype(_C,0x20)); }
inline int __cdecl iswascii(wint_t _C) {return ((unsigned)(_C) < 0x80); }


inline int __cdecl isleadbyte(int _C)
        {return (_pctype[(unsigned char)(_C)] & 0x8000); }
#line 383 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
#line 384 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 386 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"







typedef unsigned short _ino_t;      


typedef unsigned short ino_t;
#line 398 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 400 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


typedef unsigned int _dev_t;        


typedef unsigned int dev_t;
#line 407 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 409 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


typedef long _off_t;                


typedef long off_t;
#line 416 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 418 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



struct _stat {
        _dev_t st_dev;
        _ino_t st_ino;
        unsigned short st_mode;
        short st_nlink;
        short st_uid;
        short st_gid;
        _dev_t st_rdev;
        _off_t st_size;
        time_t st_atime;
        time_t st_mtime;
        time_t st_ctime;
        };



struct stat {
        _dev_t st_dev;
        _ino_t st_ino;
        unsigned short st_mode;
        short st_nlink;
        short st_uid;
        short st_gid;
        _dev_t st_rdev;
        _off_t st_size;
        time_t st_atime;
        time_t st_mtime;
        time_t st_ctime;
        };
#line 451 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



struct _stati64 {
        _dev_t st_dev;
        _ino_t st_ino;
        unsigned short st_mode;
        short st_nlink;
        short st_uid;
        short st_gid;
        _dev_t st_rdev;
        __int64 st_size;
        time_t st_atime;
        time_t st_mtime;
        time_t st_ctime;
        };

struct __stat64 {
        _dev_t st_dev;
        _ino_t st_ino;
        unsigned short st_mode;
        short st_nlink;
        short st_uid;
        short st_gid;
        _dev_t st_rdev;
        __int64 st_size;
        __time64_t st_atime;
        __time64_t st_mtime;
        __time64_t st_ctime;
        };

#line 483 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#line 486 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"






__declspec(dllimport) int __cdecl _wstat(const wchar_t *, struct _stat *);


__declspec(dllimport) int __cdecl _wstati64(const wchar_t *, struct _stati64 *);
__declspec(dllimport) int __cdecl _wstat64(const wchar_t *, struct __stat64 *);
#line 498 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#line 501 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 503 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"




__declspec(dllimport) wchar_t * __cdecl _cgetws(wchar_t *);
__declspec(dllimport) wint_t __cdecl _getwch(void);
__declspec(dllimport) wint_t __cdecl _getwche(void);
__declspec(dllimport) wint_t __cdecl _putwch(wchar_t);
__declspec(dllimport) wint_t __cdecl _ungetwch(wint_t);
__declspec(dllimport) int __cdecl _cputws(const wchar_t *);
__declspec(dllimport) int __cdecl _cwprintf(const wchar_t *, ...);
__declspec(dllimport) int __cdecl _cwscanf(const wchar_t *, ...);



#line 519 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
























































































































#line 640 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

















































































struct tm {
        int tm_sec;     
        int tm_min;     
        int tm_hour;    
        int tm_mday;    
        int tm_mon;     
        int tm_year;    
        int tm_wday;    
        int tm_yday;    
        int tm_isdst;   
        };

#line 734 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"





__declspec(dllimport) wchar_t * __cdecl _wasctime(const struct tm *);
__declspec(dllimport) wchar_t * __cdecl _wctime(const time_t *);
__declspec(dllimport) size_t __cdecl wcsftime(wchar_t *, size_t, const wchar_t *,
        const struct tm *);
__declspec(dllimport) wchar_t * __cdecl _wstrdate(wchar_t *);
__declspec(dllimport) wchar_t * __cdecl _wstrtime(wchar_t *);


__declspec(dllimport) wchar_t * __cdecl _wctime64(const __time64_t *);
#line 749 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#line 752 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"



typedef int mbstate_t;
typedef wchar_t _Wint_t;

__declspec(dllimport) wint_t __cdecl btowc(int);
__declspec(dllimport) size_t __cdecl mbrlen(const char *, size_t, mbstate_t *);
__declspec(dllimport) size_t __cdecl mbrtowc(wchar_t *, const char *, size_t, mbstate_t *);
__declspec(dllimport) size_t __cdecl mbsrtowcs(wchar_t *, const char **, size_t, mbstate_t *);

__declspec(dllimport) size_t __cdecl wcrtomb(char *, wchar_t, mbstate_t *);
__declspec(dllimport) size_t __cdecl wcsrtombs(char *, const wchar_t **, size_t, mbstate_t *);
__declspec(dllimport) int __cdecl wctob(wint_t);






#line 773 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
__declspec(dllimport) void *  __cdecl memmove(void *, const void *, size_t);
#line 775 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
void *  __cdecl memcpy(void *, const void *, size_t);

inline int fwide(FILE *, int _M)
        {return (_M); }
inline int mbsinit(const mbstate_t *_P)
        {return (_P == 0 || *_P == 0); }
inline const wchar_t *wmemchr(const wchar_t *_S, wchar_t _C, size_t _N)
        {for (; 0 < _N; ++_S, --_N)
                if (*_S == _C)
                        return (_S);
        return (0); }
inline int wmemcmp(const wchar_t *_S1, const wchar_t *_S2, size_t _N)
        {for (; 0 < _N; ++_S1, ++_S2, --_N)
                if (*_S1 != *_S2)
                        return (*_S1 < *_S2 ? -1 : +1);
        return (0); }
inline wchar_t *wmemcpy(wchar_t *_S1, const wchar_t *_S2, size_t _N)
        {
            return (wchar_t *)memcpy(_S1, _S2, _N*sizeof(wchar_t));
        }
inline wchar_t *wmemmove(wchar_t *_S1, const wchar_t *_S2, size_t _N)
        {
            return (wchar_t *)memmove(_S1, _S2, _N*sizeof(wchar_t));
        }
inline wchar_t *wmemset(wchar_t *_S, wchar_t _C, size_t _N)
        {wchar_t *_Su = _S;
        for (; 0 < _N; ++_Su, --_N)
                *_Su = _C;
        return (_S); }
}       

extern "C++" {
inline wchar_t *wmemchr(wchar_t *_S, wchar_t _C, size_t _N)
        {return ((wchar_t *)wmemchr((const wchar_t *)_S, _C, _N)); }
}

#line 812 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"


#pragma pack(pop)
#line 816 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"

#line 818 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\wchar.h"
#line 14 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cwchar"

 
namespace std {
using ::mbstate_t; using ::size_t; using ::tm; using ::wint_t;

using ::btowc; using ::fgetwc; using ::fgetws; using ::fputwc;
using ::fputws; using ::fwide; using ::fwprintf;
using ::fwscanf; using ::getwc; using ::getwchar;
using ::mbrlen; using ::mbrtowc; using ::mbsrtowcs;
using ::mbsinit; using ::putwc; using ::putwchar;
using ::swprintf; using ::swscanf; using ::ungetwc;
using ::vfwprintf; using ::vswprintf; using ::vwprintf;
using ::wcrtomb; using ::wprintf; using ::wscanf;
using ::wcsrtombs; using ::wcstol; using ::wcscat;
using ::wcschr; using ::wcscmp; using ::wcscoll;
using ::wcscpy; using ::wcscspn; using ::wcslen;
using ::wcsncat; using ::wcsncmp; using ::wcsncpy;
using ::wcspbrk; using ::wcsrchr; using ::wcsspn;
using ::wcstod; using ::wcstoul; using ::wcsstr;
using ::wcstok; using ::wcsxfrm; using ::wctob;
using ::wmemchr; using ::wmemcmp; using ::wmemcpy;
using ::wmemmove; using ::wmemset; using ::wcsftime;
}
 #line 38 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cwchar"

#line 40 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cwchar"
#line 41 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\cwchar"





#line 8 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"


#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

		

 




typedef long streamoff;
typedef int streamsize;
 #line 24 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"

extern __declspec(dllimport) fpos_t _Fpz;
extern __declspec(dllimport) const streamoff _BADOFF;

		
template<class _Statetype>
	class fpos
	{	
	typedef fpos<_Statetype> _Myt;

public:
	fpos(streamoff _Off = 0)
		: _Myoff(_Off), _Fpos(_Fpz), _Mystate(_Stz)
		{	
		}

	fpos(_Statetype _State, fpos_t _Fileposition)
		: _Myoff(0), _Fpos(_Fileposition), _Mystate(_State)
		{	
		}

	_Statetype state() const
		{	
		return (_Mystate);
		}

	void state(_Statetype _State)
		{	
		_Mystate = _State;
		}

	fpos_t seekpos() const
		{	
		return (_Fpos);
		}

	operator streamoff() const
		{	
		return (_Myoff + ((long)(_Fpos)));
		}

	streamoff operator-(const _Myt& _Right) const
		{	
		return ((streamoff)*this - (streamoff)_Right);
		}

	_Myt& operator+=(streamoff _Off)
		{	
		_Myoff += _Off;
		return (*this);
		}

	_Myt& operator-=(streamoff _Off)
		{	
		_Myoff -= _Off;
		return (*this);
		}

	_Myt operator+(streamoff _Off) const
		{	
		_Myt _Tmp = *this;
		return (_Tmp += _Off);
		}

	_Myt operator-(streamoff _Off) const
		{	
		_Myt _Tmp = *this;
		return (_Tmp -= _Off);
		}

	bool operator==(const _Myt& _Right) const
		{	
		return ((streamoff)*this == (streamoff)_Right);
		}

	bool operator!=(const _Myt& _Right) const
		{	
		return (!(*this == _Right));
		}

private:
	static _Statetype _Stz;	
	streamoff _Myoff;	
	fpos_t _Fpos;	
	_Statetype _Mystate;	
	};

	
template<class _Statetype>
	_Statetype fpos<_Statetype>::_Stz;

 

 
 

typedef fpos<mbstate_t> streampos;
typedef streampos wstreampos;

		
template<class _Elem>
	struct char_traits
	{	
	typedef _Elem char_type;
	typedef _Elem int_type;
	typedef streampos pos_type;
	typedef streamoff off_type;
	typedef mbstate_t state_type;

	static void __cdecl assign(_Elem& _Left, const _Elem& _Right)
		{	
		_Left = _Right;
		}

	static bool __cdecl eq(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left == _Right);
		}

	static bool __cdecl lt(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left < _Right);
		}

	static int __cdecl compare(const _Elem *_First1,
		const _Elem *_First2, size_t _Count)
		{	
		for (; 0 < _Count; --_Count, ++_First1, ++_First2)
			if (!eq(*_First1, *_First2))
				return (lt(*_First1, *_First2) ? -1 : +1);
		return (0);
		}

	static size_t __cdecl length(const _Elem *_First)
		{	
		size_t _Count;
		for (_Count = 0; !eq(*_First, _Elem()); ++_First)
			++_Count;
		return (_Count);
		}

	static _Elem *__cdecl copy(_Elem *_First1,
		const _Elem *_First2, size_t _Count)
		{	
		_Elem *_Next = _First1;
		for (; 0 < _Count; --_Count, ++_Next, ++_First2)
			assign(*_Next, *_First2);
		return (_First1);
		}

	static const _Elem *__cdecl find(const _Elem *_First,
		size_t _Count, const _Elem& _Ch)
		{	
		for (; 0 < _Count; --_Count, ++_First)
			if (eq(*_First, _Ch))
				return (_First);
		return (0);
		}

	static _Elem *__cdecl move(_Elem *_First1,
		const _Elem *_First2, size_t _Count)
		{	
		_Elem *_Next = _First1;
		if (_First2 < _Next && _Next < _First2 + _Count)
			for (_Next += _Count, _First2 += _Count; 0 < _Count; --_Count)
				assign(*--_Next, *--_First2);
		else
			for (; 0 < _Count; --_Count, ++_Next, ++_First2)
				assign(*_Next, *_First2);
		return (_First1);
		}

	static _Elem *__cdecl assign(_Elem *_First,
		size_t _Count, _Elem _Ch)
		{	
		_Elem *_Next = _First;
		for (; 0 < _Count; --_Count, ++_Next)
			assign(*_Next, _Ch);
		return (_First);
		}

	static _Elem __cdecl to_char_type(const int_type& _Meta)
		{	
		return (_Meta);
		}

	static int_type __cdecl to_int_type(const _Elem& _Ch)
		{	
		return (_Ch);
		}

	static bool __cdecl eq_int_type(const int_type& _Left,
		const int_type& _Right)
		{	
		return (_Left == _Right);
		}

	static int_type __cdecl eof()
		{	
		return ((int_type)(-1));
		}

	static int_type __cdecl not_eof(const int_type& _Meta)
		{	
		return (_Meta != eof() ? _Meta : !eof());
		}
	};

		
template<> struct __declspec(dllimport) char_traits<wchar_t>
	{	
	typedef wchar_t _Elem;
	typedef _Elem char_type;	
	typedef wint_t int_type;
	typedef streampos pos_type;
	typedef streamoff off_type;
	typedef mbstate_t state_type;

	static void __cdecl assign(_Elem& _Left, const _Elem& _Right)
		{	
		_Left = _Right;
		}

	static bool __cdecl eq(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left == _Right);
		}

	static bool __cdecl lt(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left < _Right);
		}

	static int __cdecl compare(const _Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return (::wmemcmp(_First1, _First2, _Count));
		}

	static size_t __cdecl length(const _Elem *_First)
		{	
		return (::wcslen(_First));
		}

	static _Elem *__cdecl copy(_Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return (::wmemcpy(_First1, _First2, _Count));
		}

	static const _Elem *__cdecl find(const _Elem *_First, size_t _Count,
		const _Elem& _Ch)
		{	
		return ((const _Elem *)::wmemchr(_First, _Ch, _Count));
		}

	static _Elem *__cdecl move(_Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return (::wmemmove(_First1, _First2, _Count));
		}

	static _Elem *__cdecl assign(_Elem *_First, size_t _Count, _Elem _Ch)
		{	
		return (::wmemset(_First, _Ch, _Count));
		}

	static _Elem __cdecl to_char_type(const int_type& _Meta)
		{	
		return (_Meta);
		}

	static int_type __cdecl to_int_type(const _Elem& _Ch)
		{	
		return (_Ch);
		}

	static bool __cdecl eq_int_type(const int_type& _Left,
		const int_type& _Right)
		{	
		return (_Left == _Right);
		}

	static int_type __cdecl eof()
		{	
		return ((wint_t)(0xFFFF));
		}

	static int_type __cdecl not_eof(const int_type& _Meta)
		{	
		return (_Meta != eof() ? _Meta : !eof());
		}
	};


		
template<> struct __declspec(dllimport) char_traits<char>
	{	
	typedef char _Elem;
	typedef _Elem char_type;
	typedef int int_type;
	typedef streampos pos_type;
	typedef streamoff off_type;
	typedef mbstate_t state_type;

	static void __cdecl assign(_Elem& _Left, const _Elem& _Right)
		{	
		_Left = _Right;
		}

	static bool __cdecl eq(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left == _Right);
		}

	static bool __cdecl lt(const _Elem& _Left, const _Elem& _Right)
		{	
		return (_Left < _Right);
		}

	static int __cdecl compare(const _Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return (::memcmp(_First1, _First2, _Count));
		}

	static size_t __cdecl length(const _Elem *_First)
		{	
		return (::strlen(_First));
		}

	static _Elem *__cdecl copy(_Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return ((_Elem *)::memcpy(_First1, _First2, _Count));
		}

	static const _Elem *__cdecl find(const _Elem *_First, size_t _Count,
		const _Elem& _Ch)
		{	
		return ((const _Elem *)::memchr(_First, _Ch, _Count));
		}

	static _Elem *__cdecl move(_Elem *_First1, const _Elem *_First2,
		size_t _Count)
		{	
		return ((_Elem *)::memmove(_First1, _First2, _Count));
		}

	static _Elem *__cdecl assign(_Elem *_First, size_t _Count, _Elem _Ch)
		{	
		return ((_Elem *)::memset(_First, _Ch, _Count));
		}

	static _Elem __cdecl to_char_type(const int_type& _Meta)
		{	
		return ((_Elem)_Meta);
		}

	static int_type __cdecl to_int_type(const _Elem& _Ch)
		{	
		return ((unsigned char)_Ch);
		}

	static bool __cdecl eq_int_type(const int_type& _Left,
		const int_type& _Right)
		{	
		return (_Left == _Right);
		}

	static int_type __cdecl eof()
		{	
		return ((-1));
		}

	static int_type __cdecl not_eof(const int_type& _Meta)
		{	
		return (_Meta != eof() ? _Meta : !eof());
		}
	};

		
template<class _Ty>
	class allocator;
class ios_base;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_ios;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class istreambuf_iterator;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class ostreambuf_iterator;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_streambuf;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_istream;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_ostream;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_iostream;
template<class _Elem,
	class _Traits = char_traits<_Elem>,
	class _Alloc = allocator<_Elem> >
	class basic_stringbuf;
template<class _Elem,
	class _Traits = char_traits<_Elem>,
	class _Alloc = allocator<_Elem> >
	class basic_istringstream;
template<class _Elem,
	class _Traits = char_traits<_Elem>,
	class _Alloc = allocator<_Elem> >
	class basic_ostringstream;
template<class _Elem,
	class _Traits = char_traits<_Elem>,
	class _Alloc = allocator<_Elem> >
	class basic_stringstream;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_filebuf;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_ifstream;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_ofstream;
template<class _Elem,
	class _Traits = char_traits<_Elem> >
	class basic_fstream;

 
template<class _Elem,
	class _InIt >
    class num_get;
template<class _Elem,
	class _OutIt >
    class num_put;
template<class _Elem>
    class collate;
 #line 469 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"

		
typedef basic_ios<char, char_traits<char> > ios;
typedef basic_streambuf<char, char_traits<char> > streambuf;
typedef basic_istream<char, char_traits<char> > istream;
typedef basic_ostream<char, char_traits<char> > ostream;
typedef basic_iostream<char, char_traits<char> > iostream;
typedef basic_stringbuf<char, char_traits<char>,
	allocator<char> > stringbuf;
typedef basic_istringstream<char, char_traits<char>,
	allocator<char> > istringstream;
typedef basic_ostringstream<char, char_traits<char>,
	allocator<char> > ostringstream;
typedef basic_stringstream<char, char_traits<char>,
	allocator<char> > stringstream;
typedef basic_filebuf<char, char_traits<char> > filebuf;
typedef basic_ifstream<char, char_traits<char> > ifstream;
typedef basic_ofstream<char, char_traits<char> > ofstream;
typedef basic_fstream<char, char_traits<char> > fstream;

		
typedef basic_ios<wchar_t, char_traits<wchar_t> > wios;
typedef basic_streambuf<wchar_t, char_traits<wchar_t> >
	wstreambuf;
typedef basic_istream<wchar_t, char_traits<wchar_t> > wistream;
typedef basic_ostream<wchar_t, char_traits<wchar_t> > wostream;
typedef basic_iostream<wchar_t, char_traits<wchar_t> > wiostream;
typedef basic_stringbuf<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> > wstringbuf;
typedef basic_istringstream<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> > wistringstream;
typedef basic_ostringstream<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> > wostringstream;
typedef basic_stringstream<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> > wstringstream;
typedef basic_filebuf<wchar_t, char_traits<wchar_t> > wfilebuf;
typedef basic_ifstream<wchar_t, char_traits<wchar_t> > wifstream;
typedef basic_ofstream<wchar_t, char_traits<wchar_t> > wofstream;
typedef basic_fstream<wchar_t, char_traits<wchar_t> > wfstream;



 
typedef num_get<char, istreambuf_iterator<char, char_traits<char> > >
	numget;
typedef num_get<wchar_t, istreambuf_iterator<wchar_t, char_traits<wchar_t> > >
	wnumget;
typedef num_put<char, ostreambuf_iterator<char, char_traits<char> > >
	numput;
typedef num_put<wchar_t, ostreambuf_iterator<wchar_t, char_traits<wchar_t> > >
	wnumput;
typedef collate<char> ncollate;
typedef collate<wchar_t> wcollate;
 #line 523 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"

}
#pragma warning(pop)
#pragma pack(pop)

#line 529 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iosfwd"





#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\utility"

#pragma pack(push,8)
#pragma warning(push,3)

namespace std {
		
template<class _Ty> inline
	void swap(_Ty& _Left, _Ty& _Right)
	{	
	_Ty _Tmp = _Left;
	_Left = _Right, _Right = _Tmp;
	}

		
template<class _Ty1,
	class _Ty2> struct pair
	{	
	typedef pair<_Ty1, _Ty2> _Myt;
	typedef _Ty1 first_type;
	typedef _Ty2 second_type;

	pair()
		: first(_Ty1()), second(_Ty2())
		{	
		}

	pair(const _Ty1& _Val1, const _Ty2& _Val2)
		: first(_Val1), second(_Val2)
		{	
		}

	template<class _Other1,
		class _Other2>
		pair(const pair<_Other1, _Other2>& _Right)
		: first(_Right.first), second(_Right.second)
		{	
		}

	void swap(_Myt& _Right)
		{	
		std::swap(first, _Right.first);
		std::swap(second, _Right.second);
		}

	_Ty1 first;	
	_Ty2 second;	
	};

		
template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator==(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (_Left.first == _Right.first && _Left.second == _Right.second);
	}

template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator!=(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (!(_Left == _Right));
	}

template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator<(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (_Left.first < _Right.first ||
		!(_Right.first < _Left.first) && _Left.second < _Right.second);
	}

template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator>(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (_Right < _Left);
	}

template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator<=(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (!(_Right < _Left));
	}

template<class _Ty1,
	class _Ty2> inline
	bool __cdecl operator>=(const pair<_Ty1, _Ty2>& _Left,
		const pair<_Ty1, _Ty2>& _Right)
	{	
	return (!(_Left < _Right));
	}

template<class _Ty1,
	class _Ty2> inline
	pair<_Ty1, _Ty2> __cdecl make_pair(_Ty1 _Val1, _Ty2 _Val2)
	{	
	return (pair<_Ty1, _Ty2>(_Val1, _Val2));
	}

template<class _Ty1,
	class _Ty2> inline
	void swap(pair<_Ty1, _Ty2>& _Left, pair<_Ty1, _Ty2>& _Right)
	{	
	_Left.swap(_Right);
	}

		
	namespace rel_ops
		{	
template<class _Ty> inline
	bool __cdecl operator!=(const _Ty& _Left, const _Ty& _Right)
	{	
	return (!(_Left == _Right));
	}

template<class _Ty> inline
	bool __cdecl operator>(const _Ty& _Left, const _Ty& _Right)
	{	
	return (_Right < _Left);
	}

template<class _Ty> inline
	bool __cdecl operator<=(const _Ty& _Left, const _Ty& _Right)
	{	
	return (!(_Right < _Left));
	}

template<class _Ty> inline
	bool __cdecl operator>=(const _Ty& _Left, const _Ty& _Right)
	{	
	return (!(_Left < _Right));
	}
		}
}
#pragma warning(pop)
#pragma pack(pop)

#line 150 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\utility"






















#line 7 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

#pragma pack(push,8)
#pragma warning(push,3)

  #pragma warning(disable:4284 4786)
namespace std {



		
struct input_iterator_tag
	{	
	};

struct output_iterator_tag
	{	
	};

struct forward_iterator_tag
	: public input_iterator_tag
	{	
	};

struct bidirectional_iterator_tag
	: public forward_iterator_tag
	{	
	};

struct random_access_iterator_tag
	: public bidirectional_iterator_tag
	{	
	};

struct _Int_iterator_tag
	{	
	};

		
struct _Nonscalar_ptr_iterator_tag
	{	
	};
struct _Scalar_ptr_iterator_tag
	{	
	};

		
template<class _Category,
	class _Ty,
	class _Diff = ptrdiff_t,
	class _Pointer = _Ty *,
	class _Reference = _Ty&>
		struct iterator
	{	
	typedef _Category iterator_category;
	typedef _Ty value_type;
	typedef _Diff difference_type;
	typedef _Diff distance_type;	
	typedef _Pointer pointer;
	typedef _Reference reference;
	};

template<class _Ty,
	class _Diff,
	class _Pointer,
	class _Reference>
	struct _Bidit
		: public iterator<bidirectional_iterator_tag, _Ty, _Diff,
			_Pointer, _Reference>
	{	
	};

template<class _Ty,
	class _Diff,
	class _Pointer,
	class _Reference>
	struct _Ranit
		: public iterator<random_access_iterator_tag, _Ty, _Diff,
			_Pointer, _Reference>
	{	
	};

struct _Outit
	: public iterator<output_iterator_tag, void, void,
		void, void>
	{	
	};

		
template<class _Iter>
	struct iterator_traits
	{	
	typedef typename _Iter::iterator_category iterator_category;
	typedef typename _Iter::value_type value_type;
	typedef typename _Iter::difference_type difference_type;
	typedef difference_type distance_type;	
	typedef typename _Iter::pointer pointer;
	typedef typename _Iter::reference reference;
	};

template<class _Ty>
	struct iterator_traits<_Ty *>
	{	
	typedef random_access_iterator_tag iterator_category;
	typedef _Ty value_type;
	typedef ptrdiff_t difference_type;
	typedef ptrdiff_t distance_type;	
	typedef _Ty *pointer;
	typedef _Ty& reference;
	};

template<class _Ty>
	struct iterator_traits<const _Ty *>
	{	
	typedef random_access_iterator_tag iterator_category;
	typedef _Ty value_type;
	typedef ptrdiff_t difference_type;
	typedef ptrdiff_t distance_type;	
	typedef const _Ty *pointer;
	typedef const _Ty& reference;
	};

template<> struct iterator_traits<_Bool>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<char>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<signed char>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<unsigned char>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

 
template<> struct iterator_traits<wchar_t>
	{	
	typedef _Int_iterator_tag iterator_category;
	};
 #line 154 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

template<> struct iterator_traits<short>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<unsigned short>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<int>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<unsigned int>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<long>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<unsigned long>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

 
template<> struct iterator_traits<__int64>
	{	
	typedef _Int_iterator_tag iterator_category;
	};

template<> struct iterator_traits<unsigned __int64>
	{	
	typedef _Int_iterator_tag iterator_category;
	};
 #line 196 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

		
template<class _Iter> inline
	typename iterator_traits<_Iter>::iterator_category
		_Iter_cat(const _Iter&)
	{	
	typename iterator_traits<_Iter>::iterator_category _Cat;
	return (_Cat);
	}


		
template<class _T1,
	class _T2> inline
	_Nonscalar_ptr_iterator_tag _Ptr_cat(_T1&, _T2&)
	{	
	_Nonscalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

template<class _Ty> inline
	_Scalar_ptr_iterator_tag _Ptr_cat(_Ty **, _Ty **)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

template<class _Ty> inline
	_Scalar_ptr_iterator_tag _Ptr_cat(_Ty **, const _Ty **)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

template<class _Ty> inline
	_Scalar_ptr_iterator_tag _Ptr_cat(_Ty *const *, _Ty **)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

template<class _Ty> inline
	_Scalar_ptr_iterator_tag _Ptr_cat(_Ty *const *, const _Ty **)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

		
inline _Scalar_ptr_iterator_tag _Ptr_cat(_Bool *, _Bool *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const _Bool *, _Bool *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(char *, char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const char *, char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(signed char *, signed char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const signed char *, signed char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(unsigned char *, unsigned char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const unsigned char *,
	unsigned char *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

 
inline _Scalar_ptr_iterator_tag _Ptr_cat(wchar_t *, wchar_t *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const wchar_t *, wchar_t *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}
 #line 307 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

inline _Scalar_ptr_iterator_tag _Ptr_cat(short *, short *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const short *, short *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(unsigned short *,
	unsigned short *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const unsigned short *,
	unsigned short *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(int *, int *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const int *, int *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(unsigned int *, unsigned int *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const unsigned int *, unsigned int *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(long *, long *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const long *, long *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(unsigned long *, unsigned long *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const unsigned long *,
	unsigned long *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(float *, float *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const float *, float *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(double *, double *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const double *, double *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(long double *, long double *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const long double *, long double *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

 
inline _Scalar_ptr_iterator_tag _Ptr_cat(__int64 *, __int64 *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const __int64 *, __int64 *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(unsigned __int64 *, unsigned __int64 *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}

inline _Scalar_ptr_iterator_tag _Ptr_cat(const unsigned __int64 *, unsigned __int64 *)
	{	
	_Scalar_ptr_iterator_tag _Cat;
	return (_Cat);
	}
 #line 444 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"


		
template<class _InIt,
	class _Diff> inline
		void _Distance2(_InIt _First, _InIt _Last, _Diff& _Off,
			input_iterator_tag)
	{	
	for (; _First != _Last; ++_First)
		++_Off;
	}

template<class _FwdIt,
	class _Diff> inline
		void _Distance2(_FwdIt _First, _FwdIt _Last, _Diff& _Off,
			forward_iterator_tag)
	{	
	for (; _First != _Last; ++_First)
		++_Off;
	}

template<class _BidIt,
	class _Diff> inline
		void _Distance2(_BidIt _First, _BidIt _Last, _Diff& _Off,
			bidirectional_iterator_tag)
	{	
	for (; _First != _Last; ++_First)
		++_Off;
	}

template<class _RanIt,
	class _Diff> inline
		void _Distance2(_RanIt _First, _RanIt _Last, _Diff& _Off,
			random_access_iterator_tag)
	{	


	_Off += _Last - _First;
	}

template<class _InIt> inline
	typename iterator_traits<_InIt>::difference_type
		distance(_InIt _First, _InIt _Last)
	{	
	typename iterator_traits<_InIt>::difference_type _Off = 0;
	_Distance2(_First, _Last, _Off, _Iter_cat(_First));
	return (_Off);
	}


template<class _InIt,
	class _Diff> inline
		void _Distance(_InIt _First, _InIt _Last, _Diff& _Off)
	{	
	_Distance2(_First, _Last, _Off, _Iter_cat(_First));
	}

		
template<class _RanIt>
	class reverse_iterator
		: public iterator<
			typename iterator_traits<_RanIt>::iterator_category,
			typename iterator_traits<_RanIt>::value_type,
			typename iterator_traits<_RanIt>::difference_type,
			typename iterator_traits<_RanIt>::pointer,
			typename iterator_traits<_RanIt>::reference>
	{	
public:
	typedef reverse_iterator<_RanIt> _Myt;
 	typedef typename iterator_traits<_RanIt>::difference_type difference_type;
	typedef typename iterator_traits<_RanIt>::pointer pointer;
	typedef typename iterator_traits<_RanIt>::reference reference;
	typedef _RanIt iterator_type;

	reverse_iterator()
		{	
		}

	explicit reverse_iterator(_RanIt _Right)
		: current(_Right)
		{	
		}

	template<class _Other>
		reverse_iterator(const reverse_iterator<_Other>& _Right)
		: current(_Right.base())
		{	
		}

	_RanIt base() const
		{	
		return (current);
		}

	reference operator*() const
		{	
		_RanIt _Tmp = current;
		return (*--_Tmp);
		}

	pointer operator->() const
		{	
		return (&**this);
		}

	_Myt& operator++()
		{	
		--current;
		return (*this);
		}

	_Myt operator++(int)
		{	
		_Myt _Tmp = *this;
		--current;
		return (_Tmp);
		}

	_Myt& operator--()
		{	
		++current;
		return (*this);
		}

	_Myt operator--(int)
		{	
		_Myt _Tmp = *this;
		++current;
		return (_Tmp);
		}

	bool _Equal(const _Myt& _Right) const
		{	
		return (current == _Right.current);
		}



	_Myt& operator+=(difference_type _Off)
		{	
		current -= _Off;
		return (*this);
		}

	_Myt operator+(difference_type _Off) const
		{	
		return (_Myt(current - _Off));
		}

	_Myt& operator-=(difference_type _Off)
		{	
		current += _Off;
		return (*this);
		}

	_Myt operator-(difference_type _Off) const
		{	
		return (_Myt(current + _Off));
		}

	reference operator[](difference_type _Off) const
		{	
		return (*(*this + _Off));
		}

	bool _Less(const _Myt& _Right) const
		{	
		return (_Right.current < current);
		}

	difference_type _Minus(const _Myt& _Right) const
		{	
		return (_Right.current - current);
		}

protected:
	_RanIt current;	
	};

		
template<class _RanIt,
	class _Diff> inline
	reverse_iterator<_RanIt> __cdecl operator+(_Diff _Off,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (_Right + _Off);
	}

template<class _RanIt> inline
	typename reverse_iterator<_RanIt>::difference_type
		__cdecl operator-(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (_Left._Minus(_Right));
	}

template<class _RanIt> inline
	bool __cdecl operator==(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (_Left._Equal(_Right));
	}

template<class _RanIt> inline
	bool __cdecl operator!=(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (!(_Left == _Right));
	}

template<class _RanIt> inline
	bool __cdecl operator<(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (_Left._Less(_Right));
	}

template<class _RanIt> inline
	bool __cdecl operator>(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (_Right < _Left);
	}

template<class _RanIt> inline
	bool __cdecl operator<=(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (!(_Right < _Left));
	}

template<class _RanIt> inline
	bool __cdecl operator>=(const reverse_iterator<_RanIt>& _Left,
		const reverse_iterator<_RanIt>& _Right)
	{	
	return (!(_Left < _Right));
	}

		
template<class _BidIt,
	class _Ty,
	class _Reference = _Ty&,
	class _Pointer = _Ty *,
	class _Diff = ptrdiff_t>
	class reverse_bidirectional_iterator
		: public _Bidit<_Ty, _Diff, _Pointer, _Reference>
	{	
public:
	typedef reverse_bidirectional_iterator<_BidIt, _Ty, _Reference,
		_Pointer, _Diff> _Myt;
	typedef _BidIt iterator_type;

	reverse_bidirectional_iterator()
		{	
		}

	explicit reverse_bidirectional_iterator(_BidIt _Right)
		: current(_Right)
		{	
		}

	_BidIt base() const
		{	
		return (current);
		}

	_Reference operator*() const
		{	
		_BidIt _Tmp = current;
		return (*--_Tmp);
		}

	_Pointer operator->() const
		{       
		_Reference _Tmp = **this;
		return (&_Tmp);
		}

	_Myt& operator++()
		{	
		--current;
		return (*this);
		}

	_Myt operator++(int)
		{	
		_Myt _Tmp = *this;
		--current;
		return (_Tmp);
		}

	_Myt& operator--()
		{	
		++current;
		return (*this);
		}

	_Myt operator--(int)
		{	
		_Myt _Tmp = *this;
		++current;
		return (_Tmp);
		}

	bool operator==(const _Myt& _Right) const
		{	
		return (current == _Right.current);
		}

	bool operator!=(const _Myt& _Right) const
		{	
		return (!(*this == _Right));
		}

protected:
	_BidIt current;	
	};

		
template<class _BidIt,
	class _BidIt2 = _BidIt>
	class _Revbidit
		: public iterator<
			typename iterator_traits<_BidIt>::iterator_category,
			typename iterator_traits<_BidIt>::value_type,
			typename iterator_traits<_BidIt>::difference_type,
			typename iterator_traits<_BidIt>::pointer,
			typename iterator_traits<_BidIt>::reference>
	{	
public:
	typedef _Revbidit<_BidIt, _BidIt2> _Myt;
	typedef typename iterator_traits<_BidIt>::difference_type _Diff;
	typedef typename iterator_traits<_BidIt>::pointer _Pointer;
	typedef typename iterator_traits<_BidIt>::reference _Reference;
	typedef _BidIt iterator_type;

	_Revbidit()
		{	
		}

	explicit _Revbidit(_BidIt _Right)
		: current(_Right)
		{	
		}

	_Revbidit(const _Revbidit<_BidIt2>& _Other)
		: current (_Other.base())
		{	
		}

	_BidIt base() const
		{	
		return (current);
		}

	_Reference operator*() const
		{	
		_BidIt _Tmp = current;
		return (*--_Tmp);
		}

	_Pointer operator->() const
		{	
		_Reference _Tmp = **this;
		return (&_Tmp);
		}

	_Myt& operator++()
		{	
		--current;
		return (*this);
		}

	_Myt operator++(int)
		{	
		_Myt _Tmp = *this;
		--current;
		return (_Tmp);
		}

	_Myt& operator--()
		{	
		++current;
		return (*this);
		}

	_Myt operator--(int)
		{	
		_Myt _Tmp = *this;
		++current;
		return (_Tmp);
		}

	bool operator==(const _Myt& _Right) const
		{	
		return (current == _Right.current);
		}

	bool operator!=(const _Myt& _Right) const
		{	
		return (!(*this == _Right));
		}

protected:
	_BidIt current;
	};

		
template<class _Elem,
	class _Traits>
	class istreambuf_iterator
		: public iterator<input_iterator_tag,
			_Elem, typename _Traits::off_type, _Elem *, _Elem&>
	{	
public:
	typedef istreambuf_iterator<_Elem, _Traits> _Myt;
	typedef _Elem char_type;
	typedef _Traits traits_type;
	typedef basic_streambuf<_Elem, _Traits> streambuf_type;
	typedef basic_istream<_Elem, _Traits> istream_type;
	typedef typename traits_type::int_type int_type;

	istreambuf_iterator(streambuf_type *_Sb = 0) throw ()
		: _Strbuf(_Sb), _Got(_Sb == 0)
		{	
		}

	istreambuf_iterator(istream_type& _Istr) throw ()
		: _Strbuf(_Istr.rdbuf()), _Got(_Istr.rdbuf() == 0)
		{	
		}

	_Elem operator*() const
		{	
		if (!_Got)
			((_Myt *)this)->_Peek();
		return (_Val);
		}

	_Myt& operator++()
		{	
		_Inc();
		return (*this);
		}

	_Myt operator++(int)
		{	
		if (!_Got)
			_Peek();
		_Myt _Tmp = *this;
		++*this;
		return (_Tmp);
		}

	bool equal(const _Myt& _Right) const
		{	
		if (!_Got)
			((_Myt *)this)->_Peek();
		if (!_Right._Got)
			((_Myt *)&_Right)->_Peek();
		return (_Strbuf == 0 && _Right._Strbuf == 0
			|| _Strbuf != 0 && _Right._Strbuf != 0);
		}

private:
	void _Inc()
		{	
		if (_Strbuf == 0
			|| traits_type::eq_int_type(traits_type::eof(),
				_Strbuf->sbumpc()))
			_Strbuf = 0, _Got = true;
		else
			_Got = false;
		}

	_Elem _Peek()
		{	
		int_type _Meta;
		if (_Strbuf == 0
			|| traits_type::eq_int_type(traits_type::eof(),
				_Meta = _Strbuf->sgetc()))
			_Strbuf = 0;
		else
			_Val = traits_type::to_char_type(_Meta);
		_Got = true;
		return (_Val);
		}

	streambuf_type *_Strbuf;	
	bool _Got;	
	_Elem _Val;	
	};

		
template<class _Elem,
	class _Traits> inline
	bool __cdecl operator==(
		const istreambuf_iterator<_Elem, _Traits>& _Left,
		const istreambuf_iterator<_Elem, _Traits>& _Right)
	{	
	return (_Left.equal(_Right));
	}

template<class _Elem,
	class _Traits> inline
	bool __cdecl operator!=(
		const istreambuf_iterator<_Elem, _Traits>& _Left,
		const istreambuf_iterator<_Elem, _Traits>& _Right)
	{	
	return (!(_Left == _Right));
	}

		
template<class _Elem,
	class _Traits>
	class ostreambuf_iterator
		: public _Outit
	{	
	typedef ostreambuf_iterator<_Elem, _Traits> _Myt;
public:
	typedef _Elem char_type;
	typedef _Traits traits_type;
	typedef basic_streambuf<_Elem, _Traits> streambuf_type;
	typedef basic_ostream<_Elem, _Traits> ostream_type;

	ostreambuf_iterator(streambuf_type *_Sb) throw ()
		: _Failed(false), _Strbuf(_Sb)
		{	
		}

	ostreambuf_iterator(ostream_type& _Ostr) throw ()
		: _Failed(false), _Strbuf(_Ostr.rdbuf())
		{	
		}

	_Myt& operator=(_Elem _Right)
		{	
		if (_Strbuf == 0
			|| traits_type::eq_int_type(_Traits::eof(),
				_Strbuf->sputc(_Right)))
			_Failed = true;
		return (*this);
		}

	_Myt& operator*()
		{	
		return (*this);
		}

	_Myt& operator++()
		{	
		return (*this);
		}

	_Myt& operator++(int)
		{	
		return (*this);
		}

	bool failed() const throw ()
		{	
		return (_Failed);
		}

private:
	bool _Failed;	
	streambuf_type *_Strbuf;	
	};



		
template<class _InIt,
	class _OutIt> inline
	_OutIt _Copy_opt(_InIt _First, _InIt _Last, _OutIt _Dest,
		_Nonscalar_ptr_iterator_tag)
	{	
	for (; _First != _Last; ++_Dest, ++_First)
		*_Dest = *_First;
	return (_Dest);
	}

template<class _InIt,
	class _OutIt> inline
	_OutIt _Copy_opt(_InIt _First, _InIt _Last, _OutIt _Dest,
		_Scalar_ptr_iterator_tag)
	{	
	ptrdiff_t _Off = _Last - _First;	
	return ((_OutIt)::memmove(&*_Dest, &*_First,
		_Off * sizeof (*_First)) + _Off);
	}

template<class _InIt,
	class _OutIt> inline
	_OutIt copy(_InIt _First, _InIt _Last, _OutIt _Dest)
	{	
	return (_Copy_opt(_First, _Last, _Dest, _Ptr_cat(_First, _Dest)));
	}

		
template<class _BidIt1,
	class _BidIt2> inline
	_BidIt2 _Copy_backward_opt(_BidIt1 _First, _BidIt1 _Last, _BidIt2 _Dest,
		_Nonscalar_ptr_iterator_tag)
	{	
	while (_First != _Last)
		*--_Dest = *--_Last;
	return (_Dest);
	}

template<class _InIt,
	class _OutIt> inline
	_OutIt _Copy_backward_opt(_InIt _First, _InIt _Last, _OutIt _Dest,
		_Scalar_ptr_iterator_tag)
	{	
	ptrdiff_t _Off = _Last - _First;	
	return ((_OutIt)memmove(&*_Dest - _Off, &*_First,
		_Off * sizeof (*_First)));
	}

template<class _BidIt1,
	class _BidIt2> inline
	_BidIt2 copy_backward(_BidIt1 _First, _BidIt1 _Last, _BidIt2 _Dest)
	{	
	return (_Copy_backward_opt(_First, _Last, _Dest,
		_Ptr_cat(_First, _Dest)));
	}

		
template<class _InIt1,
	class _InIt2> inline
	pair<_InIt1, _InIt2>
		mismatch(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2)
	{	
	for (; _First1 != _Last1 && *_First1 == *_First2; )
		++_First1, ++_First2;
	return (pair<_InIt1, _InIt2>(_First1, _First2));
	}

		
template<class _InIt1,
	class _InIt2,
	class _Pr> inline
	pair<_InIt1, _InIt2>
		mismatch(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _Pred(*_First1, *_First2); )
		++_First1, ++_First2;
	return (pair<_InIt1, _InIt2>(_First1, _First2));
	}

		
template<class _InIt1,
	class _InIt2> inline
	bool equal(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2)
	{	
	return (mismatch(_First1, _Last1, _First2).first == _Last1);
	}

inline bool equal(const char *_First1,
	const char *_Last1, const char *_First2)
	{	
	return (::memcmp(_First1, _First2, _Last1 - _First1) == 0);
	}

inline bool equal(const signed char *_First1,
	const signed char *_Last1, const signed char *_First2)
	{	
	return (::memcmp(_First1, _First2, _Last1 - _First1) == 0);
	}

inline bool equal(const unsigned char *_First1,
	const unsigned char *_Last1, const unsigned char *_First2)
	{	
	return (::memcmp(_First1, _First2, _Last1 - _First1) == 0);
	}

		
template<class _InIt1,
	class _InIt2,
	class _Pr> inline
	bool equal(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Pr _Pred)
	{	
	return (mismatch(_First1, _Last1, _First2, _Pred).first == _Last1);
	}

		
template<class _FwdIt,
	class _Ty> inline
	void fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	{	
	for (; _First != _Last; ++_First)
		*_First = _Val;
	}

inline void fill(char *_First, char *_Last, int _Val)
	{	
	::memset(_First, _Val, _Last - _First);
	}

inline void fill(signed char *_First, signed char *_Last, int _Val)
	{	
	::memset(_First, _Val, _Last - _First);
	}

inline void fill(unsigned char *_First, unsigned char *_Last, int _Val)
	{	
	::memset(_First, _Val, _Last - _First);
	}

		
template<class _OutIt,
	class _Diff,
	class _Ty> inline
	void fill_n(_OutIt _First, _Diff _Count, const _Ty& _Val)
	{	
	for (; 0 < _Count; --_Count, ++_First)
		*_First = _Val;
	}

inline void fill_n(char *_First, size_t _Count, int _Val)
	{	
	::memset(_First, _Val, _Count);
	}

inline void fill_n(signed char *_First, size_t _Count, int _Val)
	{	
	::memset(_First, _Val, _Count);
	}

inline void fill_n(unsigned char *_First, size_t _Count, int _Val)
	{	
	::memset(_First, _Val, _Count);
	}

		
template<class _InIt1,
	class _InIt2> inline
	bool lexicographical_compare(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; ++_First1, ++_First2)
		if (*_First1 < *_First2)
			return (true);
		else if (*_First2 < *_First1)
			return (false);
	return (_First1 == _Last1 && _First2 != _Last2);
	}

inline bool lexicographical_compare(
	const unsigned char *_First1, const unsigned char *_Last1,
	const unsigned char *_First2, const unsigned char *_Last2)
	{	
	ptrdiff_t _Num1 = _Last1 - _First1;
	ptrdiff_t _Num2 = _Last2 - _First2;
	int _Ans = ::memcmp(_First1, _First2, _Num1 < _Num2 ? _Num1 : _Num2);
	return (_Ans < 0 || _Ans == 0 && _Num1 < _Num2);
	}

 









#line 1214 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

		
template<class _InIt1,
	class _InIt2,
	class _Pr> inline
	bool lexicographical_compare(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; ++_First1, ++_First2)
		if (_Pred(*_First1, *_First2))
			return (true);
		else if (_Pred(*_First2, *_First1))
			return (false);
	return (_First1 == _Last1 && _First2 != _Last2);
	}

 
  
  
 #line 1234 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

 
  
  
 #line 1239 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"

		
template<class _Ty> inline
	const _Ty& (max)(const _Ty& _Left, const _Ty& _Right)
	{	
	return (_Left < _Right ? _Right : _Left);
	}

		
template<class _Ty,
	class _Pr> inline
	const _Ty& (max)(const _Ty& _Left, const _Ty& _Right, _Pr _Pred)
	{	
	return (_Pred(_Left, _Right) ? _Right : _Left);
	}

		
template<class _Ty> inline
	const _Ty& (min)(const _Ty& _Left, const _Ty& _Right)
	{	
	return (_Right < _Left ? _Right : _Left);
	}

		
template<class _Ty,
	class _Pr> inline
	const _Ty& (min)(const _Ty& _Left, const _Ty& _Right, _Pr _Pred)
	{	
	return (_Pred(_Right, _Left) ? _Right : _Left);
	}


  #pragma warning(default:4284 4786)

}
#pragma warning(pop)
#pragma pack(pop)

#line 1278 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xutility"






















#line 8 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"

#pragma pack(push,8)
#pragma warning(push,3)

 #pragma warning(disable: 4100)


 
 
 
#line 19 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"

 

 

 

 


namespace std {
		
template<class _Ty> inline
	_Ty  *_Allocate(size_t _Count, _Ty  *)
	{	
	return ((_Ty  *)operator new(_Count * sizeof (_Ty)));
	}

		
template<class _T1,
	class _T2> inline
	void _Construct(_T1  *_Ptr, const _T2& _Val)
	{	
	new ((void  *)_Ptr) _T1(_Val);
	}

		
template<class _Ty> inline
	void _Destroy(_Ty  *_Ptr)
	{	
	(_Ptr)->~_Ty();
	}

template<> inline
	void _Destroy(char  *)
	{	
	}

template<> inline
	void _Destroy(wchar_t  *)
	{	
	}


		
template<class _Ty>
	struct _Allocator_base
	{	
	typedef _Ty value_type;
	};

		
template<class _Ty>
	struct _Allocator_base<const _Ty>
	{	
	typedef _Ty value_type;
	};

		
template<class _Ty>
	class allocator
		: public _Allocator_base<_Ty>
	{	
public:
	typedef _Allocator_base<_Ty> _Mybase;
	typedef typename _Mybase::value_type value_type;


	typedef value_type  *pointer;
	typedef value_type & reference;
	typedef const value_type  *const_pointer;
	typedef const value_type & const_reference;

	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	template<class _Other>
		struct rebind
		{	
		typedef allocator<_Other> other;
		};

	pointer address(reference _Val) const
		{	
		return (&_Val);
		}

	const_pointer address(const_reference _Val) const
		{	
		return (&_Val);
		}

	allocator()
		{	
		}

	allocator(const allocator<_Ty>&)
		{	
		}

	template<class _Other>
		allocator(const allocator<_Other>&)
		{	
		}

	template<class _Other>
		allocator<_Ty>& operator=(const allocator<_Other>&)
		{	
		return (*this);
		}

	void deallocate(pointer _Ptr, size_type)
		{	
		operator delete(_Ptr);
		}

	pointer allocate(size_type _Count)
		{	
		return (_Allocate(_Count, (pointer)0));
		}

	pointer allocate(size_type _Count, const void  *)
		{	
		return (allocate(_Count));
		}

	void construct(pointer _Ptr, const _Ty& _Val)
		{	
		_Construct(_Ptr, _Val);
		}

	void destroy(pointer _Ptr)
		{	
		_Destroy(_Ptr);
		}

	size_t max_size() const
		{	
		size_t _Count = (size_t)(-1) / sizeof (_Ty);
		return (0 < _Count ? _Count : 1);
		}
	};

		
template<class _Ty,
	class _Other> inline
	bool operator==(const allocator<_Ty>&, const allocator<_Other>&)
	{	
	return (true);
	}

template<class _Ty,
	class _Other> inline
	bool operator!=(const allocator<_Ty>&, const allocator<_Other>&)
	{	
	return (false);
	}

		
template<> class __declspec(dllimport) allocator<void>
	{	
public:
	typedef void _Ty;
	typedef _Ty  *pointer;
	typedef const _Ty  *const_pointer;
	typedef _Ty value_type;

	template<class _Other>
		struct rebind
		{	
		typedef allocator<_Other> other;
		};

	allocator()
		{	
		}

	allocator(const allocator<_Ty>&)
		{	
		}

	template<class _Other>
		allocator(const allocator<_Other>&)
		{	
		}

	template<class _Other>
		allocator<_Ty>& operator=(const allocator<_Other>&)
		{	
		return (*this);
		}
	};

		
template<class _Ty,
	class _Alloc> inline
	void _Destroy_range(_Ty *_First, _Ty *_Last, _Alloc& _Al)
	{	
	_Destroy_range(_First, _Last, _Al, _Ptr_cat(_First, _Last));
	}

template<class _Ty,
	class _Alloc> inline
	void _Destroy_range(_Ty *_First, _Ty *_Last, _Alloc& _Al,
		_Nonscalar_ptr_iterator_tag)
	{	
	for (; _First != _Last; ++_First)
		_Al.destroy(_First);
	}

template<class _Ty,
	class _Alloc> inline
	void _Destroy_range(_Ty *_First, _Ty *_Last, _Alloc& _Al,
		_Scalar_ptr_iterator_tag)
	{	
	}
}

  #pragma warning(default: 4100)

#pragma warning(pop)
#pragma pack(pop)

#line 243 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xmemory"






















#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstring"

#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

  #pragma warning(disable:4251)

		
class __declspec(dllimport) _String_base
	{	
public:
	void _Xlen() const;	

	void _Xran() const;	
	};

		
template<class _Ty,
	class _Alloc>
	class _String_val
		: public _String_base
	{	
protected:
	typedef typename _Alloc::template
		rebind<_Ty>::other _Alty;

	_String_val(_Alty _Al = _Alty())
		: _Alval(_Al)
		{	
		}

	_Alty _Alval;	
	};

		
template<class _Elem,
	class _Traits = char_traits<_Elem>,
	class _Ax = allocator<_Elem> >
	class basic_string
		: public _String_val<_Elem, _Ax>
	{	
public:
	typedef basic_string<_Elem, _Traits, _Ax> _Myt;
	typedef _String_val<_Elem, _Ax> _Mybase;
	typedef typename _Mybase::_Alty _Alloc;
	typedef typename _Alloc::size_type size_type;
	typedef typename _Alloc::difference_type _Dift;
	typedef _Dift difference_type;
	typedef typename _Alloc::pointer _Tptr;
	typedef typename _Alloc::const_pointer _Ctptr;
	typedef _Tptr pointer;
	typedef _Ctptr const_pointer;
	typedef typename _Alloc::reference _Reft;
	typedef _Reft reference;
	typedef typename _Alloc::const_reference const_reference;
	typedef typename _Alloc::value_type value_type;

  
		
	class const_iterator;
	friend class const_iterator;

	class const_iterator
		: public _Ranit<_Elem, _Dift, _Ctptr, const_reference>
		{	
	public:
		typedef random_access_iterator_tag iterator_category;
		typedef _Elem value_type;
		typedef _Dift difference_type;
		typedef _Ctptr pointer;
		typedef const_reference reference;

		const_iterator()
			{	
			_Myptr = 0;
			}

 
		const_iterator(_Ctptr _Ptr)
			{	
			_Myptr = _Ptr;
			}

		const_reference operator*() const
			{	


			return (*_Myptr);
			}

		_Ctptr operator->() const
			{	
			return (&**this);
			}

		const_iterator& operator++()
			{	
			++_Myptr;
			return (*this);
			}

		const_iterator operator++(int)
			{	
			const_iterator _Tmp = *this;
			++*this;
			return (_Tmp);
			}

		const_iterator& operator--()
			{	
			--_Myptr;
			return (*this);
			}

		const_iterator operator--(int)
			{	
			const_iterator _Tmp = *this;
			--*this;
			return (_Tmp);
			}

		const_iterator& operator+=(difference_type _Off)
			{	
			_Myptr += _Off;
			return (*this);
			}

		const_iterator operator+(difference_type _Off) const
			{	
			const_iterator _Tmp = *this;
			return (_Tmp += _Off);
			}

		const_iterator& operator-=(difference_type _Off)
			{	
			return (*this += -_Off);
			}

		const_iterator operator-(difference_type _Off) const
			{	
			const_iterator _Tmp = *this;
			return (_Tmp -= _Off);
			}

		difference_type operator-(const const_iterator& _Right) const
			{	


			return (_Myptr - _Right._Myptr);
			}

		const_reference operator[](difference_type _Off) const
			{	
			return (*(*this + _Off));
			}

		bool operator==(const const_iterator& _Right) const
			{	


			return (_Myptr == _Right._Myptr);
			}

		bool operator!=(const const_iterator& _Right) const
			{	
			return (!(*this == _Right));
			}

		bool operator<(const const_iterator& _Right) const
			{	


			return (_Myptr < _Right._Myptr);
			}

		bool operator>(const const_iterator& _Right) const
			{	
			return (_Right < *this);
			}

		bool operator<=(const const_iterator& _Right) const
			{	
			return (!(_Right < *this));
			}

		bool operator>=(const const_iterator& _Right) const
			{	
			return (!(*this < _Right));
			}

		friend const_iterator operator+(difference_type _Off,
			const const_iterator& _Right)
			{	
			return (_Right + _Off);
			}


		_Ctptr _Myptr;	
		};

		
	class iterator;
	friend class iterator;

	class iterator
		: public const_iterator
		{	
	public:
		typedef random_access_iterator_tag iterator_category;
		typedef _Elem value_type;
		typedef _Dift difference_type;
		typedef _Tptr pointer;
		typedef _Reft reference;

		iterator()
			{	
			}

 
		iterator(pointer _Ptr)
			: const_iterator(_Ptr)
			{	
			}

		reference operator*() const
			{	
			return ((reference)**(const_iterator *)this);
			}

		_Tptr operator->() const
			{	
			return (&**this);
			}

		iterator& operator++()
			{	
			++this->_Myptr;
			return (*this);
			}

		iterator operator++(int)
			{	
			iterator _Tmp = *this;
			++*this;
			return (_Tmp);
			}

		iterator& operator--()
			{	
			--this->_Myptr;
			return (*this);
			}

		iterator operator--(int)
			{	
			iterator _Tmp = *this;
			--*this;
			return (_Tmp);
			}

		iterator& operator+=(difference_type _Off)
			{	
			this->_Myptr += _Off;
			return (*this);
			}

		iterator operator+(difference_type _Off) const
			{	
			iterator _Tmp = *this;
			return (_Tmp += _Off);
			}

		iterator& operator-=(difference_type _Off)
			{	
			return (*this += -_Off);
			}

		iterator operator-(difference_type _Off) const
			{	
			iterator _Tmp = *this;
			return (_Tmp -= _Off);
			}

		difference_type operator-(const const_iterator& _Right) const
			{	
			return ((const_iterator)*this - _Right);
			}

		reference operator[](difference_type _Off) const
			{	
			return (*(*this + _Off));
			}

		friend iterator operator+(difference_type _Off,
			const iterator& _Right)
			{	
			return (_Right + _Off);
			}
		};

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	basic_string()
		: _Mybase()
		{	
		_Tidy();
		}

	explicit basic_string(const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		}

	basic_string(const _Myt& _Right)
		: _Mybase(_Right._Alval)
		{	
		_Tidy();
		assign(_Right, 0, npos);
		}

	basic_string(const _Myt& _Right, size_type _Roff,
		size_type _Count = npos)
		: _Mybase()
		{	
		_Tidy();
		assign(_Right, _Roff, _Count);
		}

	basic_string(const _Myt& _Right, size_type _Roff, size_type _Count,
		const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		assign(_Right, _Roff, _Count);
		}

	basic_string(const _Elem *_Ptr, size_type _Count)
		: _Mybase()
		{	
		_Tidy();
		assign(_Ptr, _Count);
		}

	basic_string(const _Elem *_Ptr, size_type _Count, const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		assign(_Ptr, _Count);
		}

	basic_string(const _Elem *_Ptr)
		: _Mybase()
		{	
		_Tidy();
		assign(_Ptr);
		}

	basic_string(const _Elem *_Ptr, const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		assign(_Ptr);
		}

	basic_string(size_type _Count, _Elem _Ch)
		: _Mybase()
		{	
		_Tidy();
		assign(_Count, _Ch);
		}

	basic_string(size_type _Count, _Elem _Ch, const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		assign(_Count, _Ch);
		}

	template<class _It>
		basic_string(_It _First, _It _Last)
		: _Mybase()
		{	
		_Tidy();
		_Construct(_First, _Last, _Iter_cat(_First));
		}

	template<class _It>
		basic_string(_It _First, _It _Last, const _Alloc& _Al)
		: _Mybase(_Al)
		{	
		_Tidy();
		_Construct(_First, _Last, _Iter_cat(_First));
		}

	template<class _It>
		void _Construct(_It _Count,
			_It _Ch, _Int_iterator_tag)
		{	
		assign((size_type)_Count, (_Elem)_Ch);
		}

	template<class _It>
		void _Construct(_It _First,
			_It _Last, input_iterator_tag)
		{	
		try {
		for (; _First != _Last; ++_First)
			append((size_type)1, (_Elem)*_First);
		} catch (...) {
		_Tidy(true);
		throw;
		}
		}

	template<class _It>
		void _Construct(_It _First,
			_It _Last, forward_iterator_tag)
		{	
		size_type _Count = 0;
		_Distance(_First, _Last, _Count);
		reserve(_Count);

		try {
		for (; _First != _Last; ++_First)
			append((size_type)1, (_Elem)*_First);
		} catch (...) {
		_Tidy(true);
		throw;
		}
		}

	basic_string(const_pointer _First, const_pointer _Last)
		: _Mybase()
		{	
		_Tidy();
		if (_First != _Last)
			assign(&*_First, _Last - _First);
		}

	basic_string(const_iterator _First, const_iterator _Last)
		: _Mybase()
		{	
		_Tidy();
		if (_First != _Last)
			assign(&*_First, _Last - _First);
		}

	~basic_string()
		{	
		_Tidy(true);
		}

	typedef _Traits traits_type;
	typedef _Alloc allocator_type;

	static const size_type npos;	

	_Myt& operator=(const _Myt& _Right)
		{	
		return (assign(_Right));
		}

	_Myt& operator=(const _Elem *_Ptr)
		{	
		return (assign(_Ptr));
		}

	_Myt& operator=(_Elem _Ch)
		{	
		return (assign(1, _Ch));
		}

	_Myt& operator+=(const _Myt& _Right)
		{	
		return (append(_Right));
		}

	_Myt& operator+=(const _Elem *_Ptr)
		{	
		return (append(_Ptr));
		}

	_Myt& operator+=(_Elem _Ch)
		{	
		return (append((size_type)1, _Ch));
		}

	_Myt& append(const _Myt& _Right)
		{	
		return (append(_Right, 0, npos));
		}

	_Myt& append(const _Myt& _Right,
		size_type _Roff, size_type _Count)
		{	
		if (_Right.size() < _Roff)
			_String_base::_Xran();	
		size_type _Num = _Right.size() - _Roff;
		if (_Num < _Count)
			_Count = _Num;	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	

		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::copy(_Myptr() + _Mysize,
				_Right._Myptr() + _Roff, _Count);
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& append(const _Elem *_Ptr, size_type _Count)
		{	
		if (_Inside(_Ptr))
			return (append(*this, _Ptr - _Myptr(), _Count));	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	

		size_type _Num;
		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::copy(_Myptr() + _Mysize, _Ptr, _Count);
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& append(const _Elem *_Ptr)
		{	
		return (append(_Ptr, _Traits::length(_Ptr)));
		}

	_Myt& append(size_type _Count, _Elem _Ch)
		{	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	

		size_type _Num;
		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::assign(_Myptr() + _Mysize, _Count, _Ch);
			_Eos(_Num);
			}
		return (*this);
		}

	template<class _It>
		_Myt& append(_It _First, _It _Last)
		{	
		return (_Append(_First, _Last, _Iter_cat(_First)));
		}

	template<class _It>
		_Myt& _Append(_It _Count, _It _Ch, _Int_iterator_tag)
		{	
		return (append((size_type)_Count, (_Elem)_Ch));
		}

	template<class _It>
		_Myt& _Append(_It _First, _It _Last, input_iterator_tag)
		{	
		return (replace(end(), end(), _First, _Last));
		}

	_Myt& append(const_pointer _First, const_pointer _Last)
		{	
		return (replace(end(), end(), _First, _Last));
		}

	_Myt& append(const_iterator _First, const_iterator _Last)
		{	
		return (replace(end(), end(), _First, _Last));
		}

	_Myt& assign(const _Myt& _Right)
		{	
		return (assign(_Right, 0, npos));
		}

	_Myt& assign(const _Myt& _Right,
		size_type _Roff, size_type _Count)
		{	
		if (_Right.size() < _Roff)
			_String_base::_Xran();	
		size_type _Num = _Right.size() - _Roff;
		if (_Count < _Num)
			_Num = _Count;	

		if (this == &_Right)
			erase((size_type)(_Roff + _Num)), erase(0, _Roff);	
		else if (_Grow(_Num))
			{	
			_Traits::copy(_Myptr(), _Right._Myptr() + _Roff, _Num);
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& assign(const _Elem *_Ptr, size_type _Num)
		{	
		if (_Inside(_Ptr))
			return (assign(*this, _Ptr - _Myptr(), _Num));	

		if (_Grow(_Num))
			{	
			_Traits::copy(_Myptr(), _Ptr, _Num);
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& assign(const _Elem *_Ptr)
		{	
		return (assign(_Ptr, _Traits::length(_Ptr)));
		}

	_Myt& assign(size_type _Count, _Elem _Ch)
		{	
		if (_Count == npos)
			_String_base::_Xlen();	

		if (_Grow(_Count))
			{	
			_Traits::assign(_Myptr(), _Count, _Ch);
			_Eos(_Count);
			}
		return (*this);
		}

	template<class _It>
		_Myt& assign(_It _First, _It _Last)
		{	
		return (_Assign(_First, _Last, _Iter_cat(_First)));
		}

	template<class _It>
		_Myt& _Assign(_It _Count, _It _Ch, _Int_iterator_tag)
		{	
		return (assign((size_type)_Count, (_Elem)_Ch));
		}

	template<class _It>
		_Myt& _Assign(_It _First, _It _Last, input_iterator_tag)
		{	
		return (replace(begin(), end(), _First, _Last));
		}

	_Myt& assign(const_pointer _First, const_pointer _Last)
		{	
		return (replace(begin(), end(), _First, _Last));
		}

	_Myt& assign(const_iterator _First, const_iterator _Last)
		{	
		return (replace(begin(), end(), _First, _Last));
		}

	_Myt& insert(size_type _Off, const _Myt& _Right)
		{	
		return (insert(_Off, _Right, 0, npos));
		}

	_Myt& insert(size_type _Off,
		const _Myt& _Right, size_type _Roff, size_type _Count)
		{	
		if (_Mysize < _Off || _Right.size() < _Roff)
			_String_base::_Xran();	
		size_type _Num = _Right.size() - _Roff;
		if (_Num < _Count)
			_Count = _Num;	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	

		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off, _Mysize - _Off);	
			if (this == &_Right)
				_Traits::move(_Myptr() + _Off,
					_Myptr() + (_Off < _Roff ? _Roff + _Count : _Roff),
						_Count);	
			else
				_Traits::copy(_Myptr() + _Off,
					_Right._Myptr() + _Roff, _Count);	
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& insert(size_type _Off,
		const _Elem *_Ptr, size_type _Count)
		{	
		if (_Inside(_Ptr))
			return (insert(_Off, *this,
				_Ptr - _Myptr(), _Count));	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	
		size_type _Num;
		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off, _Mysize - _Off);	
			_Traits::copy(_Myptr() + _Off, _Ptr, _Count);	
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& insert(size_type _Off, const _Elem *_Ptr)
		{	
		return (insert(_Off, _Ptr, _Traits::length(_Ptr)));
		}

	_Myt& insert(size_type _Off,
		size_type _Count, _Elem _Ch)
		{	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (npos - _Mysize <= _Count)
			_String_base::_Xlen();	
		size_type _Num;
		if (0 < _Count && _Grow(_Num = _Mysize + _Count))
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off, _Mysize - _Off);	
			_Traits::assign(_Myptr() + _Off, _Count, _Ch);	
			_Eos(_Num);
			}
		return (*this);
		}

	iterator insert(iterator _Where)
		{	
		return (insert(_Where, _Elem()));
		}

	iterator insert(iterator _Where, _Elem _Ch)
		{	
		size_type _Off = _Pdif(_Where, begin());
		insert(_Off, 1, _Ch);
		return (begin() + _Off);
		}

	void insert(iterator _Where, size_type _Count, _Elem _Ch)
		{	
		size_type _Off = _Pdif(_Where, begin());
		insert(_Off, _Count, _Ch);
		}

	template<class _It>
		void insert(iterator _Where, _It _First, _It _Last)
		{	
		_Insert(_Where, _First, _Last, _Iter_cat(_First));
		}

	template<class _It>
		void _Insert(iterator _Where, _It _Count, _It _Ch,
			_Int_iterator_tag)
		{	
		insert(_Where, (size_type)_Count, (_Elem)_Ch);
		}

	template<class _It>
		void _Insert(iterator _Where, _It _First, _It _Last,
			input_iterator_tag)
		{	
		replace(_Where, _Where, _First, _Last);
		}

	void insert(iterator _Where, const_pointer _First, const_pointer _Last)
		{	
		replace(_Where, _Where, _First, _Last);
		}

	void insert(iterator _Where, const_iterator _First, const_iterator _Last)
		{	
		replace(_Where, _Where, _First, _Last);
		}

	_Myt& erase(size_type _Off = 0,
		size_type _Count = npos)
		{	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (_Mysize - _Off < _Count)
			_Count = _Mysize - _Off;	
		if (0 < _Count)
			{	
			_Traits::move(_Myptr() + _Off, _Myptr() + _Off + _Count,
				_Mysize - _Off - _Count);
			size_type _Newsize = _Mysize - _Count;
			_Eos(_Newsize);
			}
		return (*this);
		}

	iterator erase(iterator _Where)
		{	
		size_type _Count = _Pdif(_Where, begin());
		erase(_Count, 1);
		return (iterator(_Myptr() + _Count));
		}

	iterator erase(iterator _First, iterator _Last)
		{	
		size_type _Count = _Pdif(_First, begin());
		erase(_Count, _Pdif(_Last, _First));
		return (iterator(_Myptr() + _Count));
		}

	void clear()
		{	
		erase(begin(), end());
		}

	_Myt& replace(size_type _Off, size_type _N0, const _Myt& _Right)
		{	
		return (replace(_Off, _N0, _Right, 0, npos));
		}

	_Myt& replace(size_type _Off,
		size_type _N0, const _Myt& _Right, size_type _Roff, size_type _Count)
		{	
		if (_Mysize < _Off || _Right.size() < _Roff)
			_String_base::_Xran();	
		if (_Mysize - _Off < _N0)
			_N0 = _Mysize - _Off;	
		size_type _Num = _Right.size() - _Roff;
		if (_Num < _Count)
			_Count = _Num;	
		if (npos - _Count <= _Mysize - _N0)
			_String_base::_Xlen();	

		size_type _Nm = _Mysize - _N0 - _Off;	
		size_type _Newsize = _Mysize + _Count - _N0;
		if (_Mysize < _Newsize)
			_Grow(_Newsize);

		if (this != &_Right)
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
			_Traits::copy(_Myptr() + _Off,
				_Right._Myptr() + _Roff, _Count);	
			}
		else if (_Count <= _N0)
			{	
			_Traits::move(_Myptr() + _Off,
				_Myptr() + _Roff, _Count);	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
			}
		else if (_Roff <= _Off)
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
			_Traits::move(_Myptr() + _Off,
				_Myptr() + _Roff, _Count);	
			}
		else if (_Off + _N0 <= _Roff)
			{	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
			_Traits::move(_Myptr() + _Off,
				_Myptr() + (_Roff + _Count - _N0), _Count);	
			}
		else
			{	
			_Traits::move(_Myptr() + _Off,
				_Myptr() + _Roff, _N0);	
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
			_Traits::move(_Myptr() + _Off + _N0, _Myptr() + _Roff + _Count,
				_Count - _N0);	
			}

		_Eos(_Newsize);
		return (*this);
		}

	_Myt& replace(size_type _Off,
		size_type _N0, const _Elem *_Ptr, size_type _Count)
		{	
		if (_Inside(_Ptr))
			return (replace(_Off, _N0, *this,
				_Ptr - _Myptr(), _Count));	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (_Mysize - _Off < _N0)
			_N0 = _Mysize - _Off;	
		if (npos - _Count <= _Mysize - _N0)
			_String_base::_Xlen();	
		size_type _Nm = _Mysize - _N0 - _Off;

		if (_Count < _N0)
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
		size_type _Num;
		if ((0 < _Count || 0 < _N0) && _Grow(_Num = _Mysize + _Count - _N0))
			{	
			if (_N0 < _Count)
				_Traits::move(_Myptr() + _Off + _Count,
					_Myptr() + _Off + _N0, _Nm);	
			_Traits::copy(_Myptr() + _Off, _Ptr, _Count);	
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& replace(size_type _Off, size_type _N0, const _Elem *_Ptr)
		{	
		return (replace(_Off, _N0, _Ptr, _Traits::length(_Ptr)));
		}

	_Myt& replace(size_type _Off,
		size_type _N0, size_type _Count, _Elem _Ch)
		{	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (_Mysize - _Off < _N0)
			_N0 = _Mysize - _Off;	
		if (npos - _Count <= _Mysize - _N0)
			_String_base::_Xlen();	
		size_type _Nm = _Mysize - _N0 - _Off;

		if (_Count < _N0)
			_Traits::move(_Myptr() + _Off + _Count,
				_Myptr() + _Off + _N0, _Nm);	
		size_type _Num;
		if ((0 < _Count || 0 < _N0) && _Grow(_Num = _Mysize + _Count - _N0))
			{	
			if (_N0 < _Count)
				_Traits::move(_Myptr() + _Off + _Count,
					_Myptr() + _Off + _N0, _Nm);	
			_Traits::assign(_Myptr() + _Off, _Count, _Ch);	
			_Eos(_Num);
			}
		return (*this);
		}

	_Myt& replace(iterator _First, iterator _Last, const _Myt& _Right)
		{	
		return (replace(
			_Pdif(_First, begin()), _Pdif(_Last, _First), _Right));
		}

	_Myt& replace(iterator _First, iterator _Last, const _Elem *_Ptr,
		size_type _Count)
		{	
		return (replace(
			_Pdif(_First, begin()), _Pdif(_Last, _First), _Ptr, _Count));
		}

	_Myt& replace(iterator _First, iterator _Last, const _Elem *_Ptr)
		{	
		return (replace(
			_Pdif(_First, begin()), _Pdif(_Last, _First), _Ptr));
		}

	_Myt& replace(iterator _First, iterator _Last,
		size_type _Count, _Elem _Ch)
		{	
		return (replace(
			_Pdif(_First, begin()), _Pdif(_Last, _First), _Count, _Ch));
		}

	template<class _It>
		_Myt& replace(iterator _First, iterator _Last,
			_It _First2, _It _Last2)
		{	
		return (_Replace(_First, _Last,
			_First2, _Last2, _Iter_cat(_First2)));
		}

	template<class _It>
		_Myt& _Replace(iterator _First, iterator _Last,
			_It _Count, _It _Ch, _Int_iterator_tag)
		{	
		return (replace(_First, _Last, (size_type)_Count, (_Elem)_Ch));
		}

	template<class _It>
		_Myt& _Replace(iterator _First, iterator _Last,
			_It _First2, _It _Last2, input_iterator_tag)
		{	
		_Myt _Right(_First2, _Last2);
		replace(_First, _Last, _Right);
		return (*this);
		}

	_Myt& replace(iterator _First, iterator _Last,
		const_pointer _First2, const_pointer _Last2)
		{	
		if (_First2 == _Last2)
			erase(_Pdif(_First, begin()), _Pdif(_Last, _First));
		else
			replace(_Pdif(_First, begin()), _Pdif(_Last, _First),
				&*_First2, _Last2 - _First2);
		return (*this);
		}

	_Myt& replace(iterator _First, iterator _Last,
		const_iterator _First2, const_iterator _Last2)
		{	
		if (_First2 == _Last2)
			erase(_Pdif(_First, begin()), _Pdif(_Last, _First));
		else
			replace(_Pdif(_First, begin()), _Pdif(_Last, _First),
				&*_First2, _Last2 - _First2);
		return (*this);
		}

	iterator begin()
		{	
		return (iterator(_Myptr()));
		}

	const_iterator begin() const
		{	
		return (const_iterator(_Myptr()));
		}

	iterator end()
		{	
		return (iterator(_Myptr() + _Mysize));
		}

	const_iterator end() const
		{	
		return (const_iterator(_Myptr() + _Mysize));
		}

	reverse_iterator rbegin()
		{	
		return (reverse_iterator(end()));
		}

	const_reverse_iterator rbegin() const
		{	
		return (const_reverse_iterator(end()));
		}

	reverse_iterator rend()
		{	
		return (reverse_iterator(begin()));
		}

	const_reverse_iterator rend() const
		{	
		return (const_reverse_iterator(begin()));
		}

	reference at(size_type _Off)
		{	
		if (_Mysize <= _Off)
			_String_base::_Xran();	
		return (_Myptr()[_Off]);
		}

	const_reference at(size_type _Off) const
		{	
		if (_Mysize <= _Off)
			_String_base::_Xran();	
		return (_Myptr()[_Off]);
		}

	reference operator[](size_type _Off)
		{	
		return (_Myptr()[_Off]);
		}

	const_reference operator[](size_type _Off) const
		{	
		return (_Myptr()[_Off]);
		}

	void push_back(_Elem _Ch)
		{	
		insert(end(), _Ch);
		}

	const _Elem *c_str() const
		{	
		return (_Myptr());
		}

	const _Elem *data() const
		{	
		return (c_str());
		}

	size_type length() const
		{	
		return (_Mysize);
		}

	size_type size() const
		{	
		return (_Mysize);
		}

	size_type max_size() const
		{	
		size_type _Num = _Mybase::_Alval.max_size();
		return (_Num <= 1 ? 1 : _Num - 1);
		}

	void resize(size_type _Newsize)
		{	
		resize(_Newsize, _Elem());
		}

	void resize(size_type _Newsize, _Elem _Ch)
		{	
		if (_Newsize <= _Mysize)
			erase(_Newsize);
		else
			append(_Newsize - _Mysize, _Ch);
		}

	size_type capacity() const
		{	
		return (_Myres);
		}

	void reserve(size_type _Newcap = 0)
		{	
		if (_Mysize <= _Newcap && _Myres != _Newcap)
			{	
			size_type _Size = _Mysize;
			if (_Grow(_Newcap, true))
				_Eos(_Size);
			}
		}

	bool empty() const
		{	
		return (_Mysize == 0);
		}

	size_type copy(_Elem *_Ptr,
		size_type _Count, size_type _Off = 0) const
		{	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (_Mysize - _Off < _Count)
			_Count = _Mysize - _Off;
		_Traits::copy(_Ptr, _Myptr() + _Off, _Count);
		return (_Count);
		}

	void swap(_Myt& _Right)
		{	
		if (_Mybase::_Alval == _Right._Alval)
			{	
			_Bxty _Tbx = _Bx;
			_Bx = _Right._Bx, _Right._Bx = _Tbx;

			size_type _Tlen = _Mysize;
			_Mysize = _Right._Mysize, _Right._Mysize = _Tlen;

			size_type _Tres = _Myres;
			_Myres = _Right._Myres, _Right._Myres = _Tres;
			}
		else
			{	
			_Myt _Tmp = *this; *this = _Right, _Right = _Tmp;
			}
		}

	size_type find(const _Myt& _Right, size_type _Off = 0) const
		{	
		return (find(_Right._Myptr(), _Off, _Right.size()));
		}

	size_type find(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (_Count == 0 && _Off <= _Mysize)
			return (_Off);	

		size_type _Nm;
		if (_Off < _Mysize && _Count <= (_Nm = _Mysize - _Off))
			{	
			const _Elem *_Uptr, *_Vptr;
			for (_Nm -= _Count - 1, _Vptr = _Myptr() + _Off;
				(_Uptr = _Traits::find(_Vptr, _Nm, *_Ptr)) != 0;
				_Nm -= _Uptr - _Vptr + 1, _Vptr = _Uptr + 1)
				if (_Traits::compare(_Uptr, _Ptr, _Count) == 0)
					return (_Uptr - _Myptr());	
			}

		return (npos);	
		}

	size_type find(const _Elem *_Ptr, size_type _Off = 0) const
		{	
		return (find(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type find(_Elem _Ch, size_type _Off = 0) const
		{	
		return (find((const _Elem *)&_Ch, _Off, 1));
		}

	size_type rfind(const _Myt& _Right, size_type _Off = npos) const
		{	
		return (rfind(_Right._Myptr(), _Off, _Right.size()));
		}

	size_type rfind(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (_Count == 0)
			return (_Off < _Mysize ? _Off : _Mysize);	
		if (_Count <= _Mysize)
			{	
			const _Elem *_Uptr = _Myptr() +
				(_Off < _Mysize - _Count ? _Off : _Mysize - _Count);
			for (; ; --_Uptr)
				if (_Traits::eq(*_Uptr, *_Ptr)
					&& _Traits::compare(_Uptr, _Ptr, _Count) == 0)
					return (_Uptr - _Myptr());	
				else if (_Uptr == _Myptr())
					break;	
			}

		return (npos);	
		}

	size_type rfind(const _Elem *_Ptr, size_type _Off = npos) const
		{	
		return (rfind(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type rfind(_Elem _Ch, size_type _Off = npos) const
		{	
		return (rfind((const _Elem *)&_Ch, _Off, 1));
		}

	size_type find_first_of(const _Myt& _Right,
		size_type _Off = 0) const
		{	
		return (find_first_of(_Right._Myptr(), _Off, _Right.size()));
		}

	size_type find_first_of(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (0 < _Count && _Off < _Mysize)
			{	
			const _Elem *const _Vptr = _Myptr() + _Mysize;
			for (const _Elem *_Uptr = _Myptr() + _Off; _Uptr < _Vptr; ++_Uptr)
				if (_Traits::find(_Ptr, _Count, *_Uptr) != 0)
					return (_Uptr - _Myptr());	
			}

		return (npos);	
		}

	size_type find_first_of(const _Elem *_Ptr, size_type _Off = 0) const
		{	
		return (find_first_of(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type find_first_of(_Elem _Ch, size_type _Off = 0) const
		{	
		return (find((const _Elem *)&_Ch, _Off, 1));
		}

	size_type find_last_of(const _Myt& _Right,
		size_type _Off = npos) const
		{	
		return (find_last_of(_Right._Myptr(), _Off, _Right.size()));
		}

	size_type find_last_of(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (0 < _Count && 0 < _Mysize)
			for (const _Elem *_Uptr = _Myptr()
				+ (_Off < _Mysize ? _Off : _Mysize - 1); ; --_Uptr)
				if (_Traits::find(_Ptr, _Count, *_Uptr) != 0)
					return (_Uptr - _Myptr());	
				else if (_Uptr == _Myptr())
					break;	

		return (npos);	
		}

	size_type find_last_of(const _Elem *_Ptr,
		size_type _Off = npos) const
		{	
		return (find_last_of(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type find_last_of(_Elem _Ch, size_type _Off = npos) const
		{	
		return (rfind((const _Elem *)&_Ch, _Off, 1));
		}

	size_type find_first_not_of(const _Myt& _Right,
		size_type _Off = 0) const
		{	
		return (find_first_not_of(_Right._Myptr(), _Off,
			_Right.size()));
		}

	size_type find_first_not_of(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (_Off < _Mysize)
			{	
			const _Elem *const _Vptr = _Myptr() + _Mysize;
			for (const _Elem *_Uptr = _Myptr() + _Off; _Uptr < _Vptr; ++_Uptr)
				if (_Traits::find(_Ptr, _Count, *_Uptr) == 0)
					return (_Uptr - _Myptr());
			}
		return (npos);
		}

	size_type find_first_not_of(const _Elem *_Ptr,
		size_type _Off = 0) const
		{	
		return (find_first_not_of(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type find_first_not_of(_Elem _Ch, size_type _Off = 0) const
		{	
		return (find_first_not_of((const _Elem *)&_Ch, _Off, 1));
		}

	size_type find_last_not_of(const _Myt& _Right,
		size_type _Off = npos) const
		{	
		return (find_last_not_of(_Right._Myptr(), _Off, _Right.size()));
		}

	size_type find_last_not_of(const _Elem *_Ptr,
		size_type _Off, size_type _Count) const
		{	
		if (0 < _Mysize)
			for (const _Elem *_Uptr = _Myptr()
				+ (_Off < _Mysize ? _Off : _Mysize - 1); ; --_Uptr)
				if (_Traits::find(_Ptr, _Count, *_Uptr) == 0)
					return (_Uptr - _Myptr());
				else if (_Uptr == _Myptr())
					break;
		return (npos);
		}

	size_type find_last_not_of(const _Elem *_Ptr,
		size_type _Off = npos) const
		{	
		return (find_last_not_of(_Ptr, _Off, _Traits::length(_Ptr)));
		}

	size_type find_last_not_of(_Elem _Ch, size_type _Off = npos) const
		{	
		return (find_last_not_of((const _Elem *)&_Ch, _Off, 1));
		}

	_Myt substr(size_type _Off = 0, size_type _Count = npos) const
		{	
		return (_Myt(*this, _Off, _Count));
		}

	int compare(const _Myt& _Right) const
		{	
		return (compare(0, _Mysize, _Right._Myptr(), _Right.size()));
		}

	int compare(size_type _Off, size_type _N0,
		const _Myt& _Right) const
		{	
		return (compare(_Off, _N0, _Right, 0, npos));
		}

	int compare(size_type _Off,
		size_type _N0, const _Myt& _Right,
		size_type _Roff, size_type _Count) const
		{	
		if (_Right.size() < _Roff)
			_String_base::_Xran();	
		if (_Right._Mysize - _Roff < _Count)
			_Count = _Right._Mysize - _Roff;	
		return (compare(_Off, _N0, _Right._Myptr() + _Roff, _Count));
		}

	int compare(const _Elem *_Ptr) const
		{	
		return (compare(0, _Mysize, _Ptr, _Traits::length(_Ptr)));
		}

	int compare(size_type _Off, size_type _N0, const _Elem *_Ptr) const
		{	
		return (compare(_Off, _N0, _Ptr, _Traits::length(_Ptr)));
		}

	int compare(size_type _Off,
		size_type _N0, const _Elem *_Ptr, size_type _Count) const
		{	
		if (_Mysize < _Off)
			_String_base::_Xran();	
		if (_Mysize - _Off < _N0)
			_N0 = _Mysize - _Off;	

		size_type _Ans = _N0 == 0 ? 0
			: _Traits::compare(_Myptr() + _Off, _Ptr,
				_N0 < _Count ? _N0 : _Count);
		return (_Ans != 0 ? (int)_Ans : _N0 < _Count ? -1
			: _N0 == _Count ? 0 : +1);
		}

	allocator_type get_allocator() const
		{	
		return (_Mybase::_Alval);
		}

	enum
		{	
		_BUF_SIZE = 16 / sizeof (_Elem) < 1 ? 1
			: 16 / sizeof(_Elem)};

protected:
	enum
		{	
		_ALLOC_MASK = sizeof (_Elem) <= 1 ? 15
			: sizeof (_Elem) <= 2 ? 7
			: sizeof (_Elem) <= 4 ? 3
			: sizeof (_Elem) <= 8 ? 1 : 0};

	void _Copy(size_type _Newsize, size_type _Oldlen)
		{	
		size_type _Newres = _Newsize | _ALLOC_MASK;
		if (max_size() < _Newres)
			_Newres = _Newsize;	
		else if (_Newres / 3 < _Myres / 2
			&& _Myres <= max_size() - _Myres / 2)
			_Newres = _Myres + _Myres / 2;	
		_Elem *_Ptr;

		try {
			_Ptr = _Mybase::_Alval.allocate(_Newres + 1);
		} catch (...) {
			_Newres = _Newsize;	
			try {
				_Ptr = _Mybase::_Alval.allocate(_Newres + 1);
			} catch (...) {
			_Tidy(true);	
			throw;
			}
		}

		if (0 < _Oldlen)
			_Traits::copy(_Ptr, _Myptr(), _Oldlen);	
		_Tidy(true);
		_Bx._Ptr = _Ptr;
		_Myres = _Newres;
		_Eos(_Oldlen);
		}

	void _Eos(size_type _Newsize)
		{	
		_Traits::assign(_Myptr()[_Mysize = _Newsize], _Elem());
		}

	bool _Grow(size_type _Newsize,
		bool _Trim = false)
		{	
		if (max_size() < _Newsize)
			_String_base::_Xlen();	
		if (_Myres < _Newsize)
			_Copy(_Newsize, _Mysize);	
		else if (_Trim && _Newsize < _BUF_SIZE)
			_Tidy(true,	
				_Newsize < _Mysize ? _Newsize : _Mysize);
		else if (_Newsize == 0)
			_Eos(0);	
		return (0 < _Newsize);	
		}

	bool _Inside(const _Elem *_Ptr)
		{	
		if (_Ptr < _Myptr() || _Myptr() + _Mysize <= _Ptr)
			return (false);	
		else
			return (true);
		}

	static size_type __cdecl _Pdif(const_iterator _P2,
		const_iterator _P1)
		{	
		return ((_P2)._Myptr == 0 ? 0 : _P2 - _P1);
		}

	void _Tidy(bool _Built = false,
		size_type _Newsize = 0)
		{	
		if (!_Built)
			;
		else if (_BUF_SIZE <= _Myres)
			{	
			_Elem *_Ptr = _Bx._Ptr;
			if (0 < _Newsize)
				_Traits::copy(_Bx._Buf, _Ptr, _Newsize);
			_Mybase::_Alval.deallocate(_Ptr, _Myres + 1);
			}
		_Myres = _BUF_SIZE - 1;
		_Eos(_Newsize);
		}

	union _Bxty
		{	
		_Elem _Buf[_BUF_SIZE];
		_Elem *_Ptr;
		} _Bx;

public:
	_Elem *_Myptr()
		{	
		return (_BUF_SIZE <= _Myres ? _Bx._Ptr : _Bx._Buf);
		}

	const _Elem *_Myptr() const
		{	
		return (_BUF_SIZE <= _Myres ? _Bx._Ptr : _Bx._Buf);
		}

	size_type _Mysize;	
	size_type _Myres;	
	};

		
template<class _Elem,
	class _Traits,
	class _Alloc>
	const typename basic_string<_Elem, _Traits, _Alloc>::size_type
		basic_string<_Elem, _Traits, _Alloc>::npos =
			(basic_string<_Elem, _Traits, _Alloc>::size_type)(-1);

template<class _Elem,
	class _Traits,
	class _Alloc> inline
	void swap(basic_string<_Elem, _Traits, _Alloc>& _Left,
		basic_string<_Elem, _Traits, _Alloc>& _Right)
	{	
	_Left.swap(_Right);
	}

typedef basic_string<char, char_traits<char>, allocator<char> >
	string;
typedef basic_string<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> > wstring;

 

template class __declspec(dllimport) basic_string<char, char_traits<char>,
	allocator<char> >;
template class __declspec(dllimport) basic_string<wchar_t, char_traits<wchar_t>,
	allocator<wchar_t> >;




 #line 1577 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstring"
}
 #pragma warning(default: 4251)
#pragma warning(pop)
#pragma pack(pop)

#line 1583 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\xstring"





#line 7 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

		
class logic_error
	: public std:: exception
	{	
public:
	explicit logic_error(const string& _Message)
		: _Str(_Message)
		{	
		}

	virtual ~logic_error()
		{}	

	virtual const char *what() const throw ()
		{	
		return (_Str.c_str());
		}

 





#line 37 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

private:
	string _Str;	
	};

		
class domain_error
	: public logic_error
	{	
public:
	explicit domain_error(const string& _Message)
		: logic_error(_Message)
		{	
		}

	virtual ~domain_error()
		{}	

 





#line 62 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class invalid_argument
	: public logic_error
	{	
public:
	explicit invalid_argument(const string& _Message)
		: logic_error(_Message)
		{	
		}

	virtual ~invalid_argument()
		{}	

 





#line 85 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class length_error
	: public logic_error
	{	
public:
	explicit length_error(const string& _Message)
		: logic_error(_Message)
		{	
		}

	virtual ~length_error()
		{}	

 





#line 108 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class out_of_range
	: public logic_error
	{	
public:
	explicit out_of_range(const string& _Message)
		: logic_error(_Message)
		{	
		}

	virtual ~out_of_range()
		{}	

 





#line 131 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class runtime_error
	: public std:: exception
	{	
public:
	explicit runtime_error(const string& _Message)
		: _Str(_Message)
		{	
		}

	virtual ~runtime_error()
		{}	

	virtual const char *what() const throw ()
		{	
		return (_Str.c_str());
		}

 





#line 159 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

private:
	string _Str;	
	};

		
class overflow_error
	: public runtime_error
	{	
public:
	explicit overflow_error(const string& _Message)
		: runtime_error(_Message)
		{	
		}

	virtual ~overflow_error()
		{}	

 





#line 184 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class underflow_error
	: public runtime_error
	{	
public:
	explicit underflow_error(const string& _Message)
		: runtime_error(_Message)
		{	
		}

	virtual ~underflow_error()
		{}	

 





#line 207 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};

		
class range_error
	: public runtime_error
	{	
public:
	explicit range_error(const string& _Message)
		: runtime_error(_Message)
		{	
		}

	virtual ~range_error()
		{}	

 





#line 230 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"

	};
}
#pragma warning(pop)
#pragma pack(pop)

#line 237 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\stdexcept"





#line 29 "c:\\boost\\boost/array.hpp"
#line 1 "c:\\boost\\boost/assert.hpp"




















#line 22 "c:\\boost\\boost/assert.hpp"












#line 35 "c:\\boost\\boost/assert.hpp"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"















#line 17 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"



















#line 37 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"










extern "C" {
#line 49 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"

__declspec(dllimport) void __cdecl _assert(const char *, const char *, unsigned);


}
#line 55 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"



#line 59 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\assert.h"
#line 36 "c:\\boost\\boost/assert.hpp"

#line 38 "c:\\boost\\boost/assert.hpp"
#line 30 "c:\\boost\\boost/array.hpp"


#line 1 "c:\\boost\\boost/detail/iterator.hpp"




















































#line 1 "c:\\boost\\boost/config.hpp"






















#line 24 "c:\\boost\\boost/config.hpp"


#line 1 "c:\\boost\\boost/config/user.hpp"



























































































 











 







 









 


#line 27 "c:\\boost\\boost/config.hpp"
#line 28 "c:\\boost\\boost/config.hpp"



#line 1 "c:\\boost\\boost/config/select_compiler_config.hpp"


















#line 20 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 24 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 28 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 32 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 36 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 40 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 44 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 48 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 52 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 56 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 60 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 64 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 68 "c:\\boost\\boost/config/select_compiler_config.hpp"



#line 72 "c:\\boost\\boost/config/select_compiler_config.hpp"











#line 84 "c:\\boost\\boost/config/select_compiler_config.hpp"
#line 32 "c:\\boost\\boost/config.hpp"
#line 33 "c:\\boost\\boost/config.hpp"


#line 1 "c:\\boost\\boost/config/compiler/visualc.hpp"

















#pragma warning( disable : 4503 ) 








#line 28 "c:\\boost\\boost/config/compiler/visualc.hpp"



































#line 64 "c:\\boost\\boost/config/compiler/visualc.hpp"



#line 68 "c:\\boost\\boost/config/compiler/visualc.hpp"



#line 72 "c:\\boost\\boost/config/compiler/visualc.hpp"





















#line 94 "c:\\boost\\boost/config/compiler/visualc.hpp"


#line 97 "c:\\boost\\boost/config/compiler/visualc.hpp"

















#line 115 "c:\\boost\\boost/config/compiler/visualc.hpp"


#line 118 "c:\\boost\\boost/config/compiler/visualc.hpp"



#line 122 "c:\\boost\\boost/config/compiler/visualc.hpp"

#line 124 "c:\\boost\\boost/config/compiler/visualc.hpp"





#line 130 "c:\\boost\\boost/config/compiler/visualc.hpp"








#line 139 "c:\\boost\\boost/config/compiler/visualc.hpp"








#line 148 "c:\\boost\\boost/config/compiler/visualc.hpp"
#line 36 "c:\\boost\\boost/config.hpp"
#line 37 "c:\\boost\\boost/config.hpp"



#line 1 "c:\\boost\\boost/config/select_stdlib_config.hpp"



























#line 29 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 33 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 37 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 41 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 45 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 49 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 53 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 57 "c:\\boost\\boost/config/select_stdlib_config.hpp"








#line 66 "c:\\boost\\boost/config/select_stdlib_config.hpp"



#line 41 "c:\\boost\\boost/config.hpp"
#line 42 "c:\\boost\\boost/config.hpp"


#line 1 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"


















#line 20 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"



   
   


#line 28 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"


#line 31 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"









#line 41 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"



#line 45 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"

















#line 63 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"








#line 72 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"







#line 80 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"





#line 86 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"





#line 92 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"





#line 98 "c:\\boost\\boost/config/stdlib/dinkumware.hpp"









#line 45 "c:\\boost\\boost/config.hpp"
#line 46 "c:\\boost\\boost/config.hpp"



#line 1 "c:\\boost\\boost/config/select_platform_config.hpp"



















#line 21 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 25 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 29 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 33 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 37 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 41 "c:\\boost\\boost/config/select_platform_config.hpp"










































#line 84 "c:\\boost\\boost/config/select_platform_config.hpp"



#line 50 "c:\\boost\\boost/config.hpp"
#line 51 "c:\\boost\\boost/config.hpp"


#line 1 "c:\\boost\\boost/config/platform/win32.hpp"















#line 17 "c:\\boost\\boost/config/platform/win32.hpp"



#line 21 "c:\\boost\\boost/config/platform/win32.hpp"




#line 26 "c:\\boost\\boost/config/platform/win32.hpp"

















#line 44 "c:\\boost\\boost/config/platform/win32.hpp"






#line 51 "c:\\boost\\boost/config/platform/win32.hpp"
#line 54 "c:\\boost\\boost/config.hpp"
#line 55 "c:\\boost\\boost/config.hpp"


#line 1 "c:\\boost\\boost/config/suffix.hpp"



































#line 39 "c:\\boost\\boost/config/suffix.hpp"


#line 42 "c:\\boost\\boost/config/suffix.hpp"






#line 49 "c:\\boost\\boost/config/suffix.hpp"














#line 65 "c:\\boost\\boost/config/suffix.hpp"









#line 76 "c:\\boost\\boost/config/suffix.hpp"







#line 84 "c:\\boost\\boost/config/suffix.hpp"







#line 92 "c:\\boost\\boost/config/suffix.hpp"







#line 101 "c:\\boost\\boost/config/suffix.hpp"






#line 109 "c:\\boost\\boost/config/suffix.hpp"






#line 117 "c:\\boost\\boost/config/suffix.hpp"






#line 125 "c:\\boost\\boost/config/suffix.hpp"







#line 135 "c:\\boost\\boost/config/suffix.hpp"







#line 145 "c:\\boost\\boost/config/suffix.hpp"






#line 152 "c:\\boost\\boost/config/suffix.hpp"






#line 159 "c:\\boost\\boost/config/suffix.hpp"






#line 166 "c:\\boost\\boost/config/suffix.hpp"






#line 173 "c:\\boost\\boost/config/suffix.hpp"






#line 180 "c:\\boost\\boost/config/suffix.hpp"






#line 187 "c:\\boost\\boost/config/suffix.hpp"






#line 194 "c:\\boost\\boost/config/suffix.hpp"








#line 204 "c:\\boost\\boost/config/suffix.hpp"









#line 215 "c:\\boost\\boost/config/suffix.hpp"






#line 222 "c:\\boost\\boost/config/suffix.hpp"






#line 231 "c:\\boost\\boost/config/suffix.hpp"


















#line 250 "c:\\boost\\boost/config/suffix.hpp"







#line 258 "c:\\boost\\boost/config/suffix.hpp"



#line 262 "c:\\boost\\boost/config/suffix.hpp"






#line 269 "c:\\boost\\boost/config/suffix.hpp"



#line 273 "c:\\boost\\boost/config/suffix.hpp"





















#line 295 "c:\\boost\\boost/config/suffix.hpp"



#line 299 "c:\\boost\\boost/config/suffix.hpp"




























#line 328 "c:\\boost\\boost/config/suffix.hpp"

























#line 354 "c:\\boost\\boost/config/suffix.hpp"


#line 357 "c:\\boost\\boost/config/suffix.hpp"















#line 373 "c:\\boost\\boost/config/suffix.hpp"










#line 384 "c:\\boost\\boost/config/suffix.hpp"
















#line 401 "c:\\boost\\boost/config/suffix.hpp"







namespace boost{




   typedef long long long_long_type;
   typedef unsigned long long ulong_long_type;
#line 416 "c:\\boost\\boost/config/suffix.hpp"
}
#line 418 "c:\\boost\\boost/config/suffix.hpp"































































#line 482 "c:\\boost\\boost/config/suffix.hpp"














#line 497 "c:\\boost\\boost/config/suffix.hpp"











































#line 541 "c:\\boost\\boost/config/suffix.hpp"



#line 58 "c:\\boost\\boost/config.hpp"

#line 60 "c:\\boost\\boost/config.hpp"











#line 54 "c:\\boost\\boost/detail/iterator.hpp"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iterator"

#pragma once




#pragma pack(push,8)
#pragma warning(push,3)

namespace std {

		
template<class _Container>
	class back_insert_iterator
		: public _Outit
	{	
public:
	typedef _Container container_type;
	typedef typename _Container::reference reference;

	explicit back_insert_iterator(_Container& _Cont)
		: container(&_Cont)
		{	
		}

	back_insert_iterator<_Container>& operator=(
		typename _Container::const_reference _Val)
		{	
		container->push_back(_Val);
		return (*this);
		}

	back_insert_iterator<_Container>& operator*()
		{	
		return (*this);
		}

	back_insert_iterator<_Container>& operator++()
		{	
		return (*this);
		}

	back_insert_iterator<_Container> operator++(int)
		{	
		return (*this);
		}

protected:
	_Container *container;	
	};

		
template<class _Container> inline
	back_insert_iterator<_Container> back_inserter(_Container& _Cont)
	{	
	return (std::back_insert_iterator<_Container>(_Cont));
	}

		
template<class _Container>
	class front_insert_iterator
		: public _Outit
	{	
public:
	typedef _Container container_type;
	typedef typename _Container::reference reference;

	explicit front_insert_iterator(_Container& _Cont)
		: container(&_Cont)
		{	
		}

	front_insert_iterator<_Container>& operator=(
		typename _Container::const_reference _Val)
		{	
		container->push_front(_Val);
		return (*this);
		}

	front_insert_iterator<_Container>& operator*()
		{	
		return (*this);
		}

	front_insert_iterator<_Container>& operator++()
		{	
		return (*this);
		}

	front_insert_iterator<_Container> operator++(int)
		{	
		return (*this);
		}

protected:
	_Container *container;	
	};

		
template<class _Container> inline
	front_insert_iterator<_Container> front_inserter(_Container& _Cont)
	{	
	return (std::front_insert_iterator<_Container>(_Cont));
	}

		
template<class _Container>
	class insert_iterator
		: public _Outit
	{	
public:
	typedef _Container container_type;
	typedef typename _Container::reference reference;

	insert_iterator(_Container& _Cont, typename _Container::iterator _Where)
		: container(&_Cont), iter(_Where)
		{	
		}

	insert_iterator<_Container>& operator=(
		typename _Container::const_reference _Val)
		{	
		iter = container->insert(iter, _Val);
		++iter;
		return (*this);
		}

	insert_iterator<_Container>& operator*()
		{	
		return (*this);
		}

	insert_iterator<_Container>& operator++()
		{	
		return (*this);
		}

	insert_iterator<_Container>& operator++(int)
		{	
		return (*this);
		}

protected:
	_Container *container;	
	typename _Container::iterator iter;	
	};

		
template<class _Container,
	class _Iter> inline
	insert_iterator<_Container> inserter(_Container& _Cont, _Iter _Where)
	{	
	return (std::insert_iterator<_Container>(_Cont, _Where));
	}

		
template<class _Ty,
	class _Elem = char,
	class _Traits = char_traits<_Elem>,
	class _Diff = ptrdiff_t>
	class istream_iterator
		: public iterator<input_iterator_tag, _Ty, _Diff,
			const _Ty *, const _Ty&>
	{	
public:
	typedef istream_iterator<_Ty, _Elem, _Traits, _Diff> _Myt;
	typedef _Elem char_type;
	typedef _Traits traits_type;
	typedef basic_istream<_Elem, _Traits> istream_type;

	istream_iterator()
		: _Myistr(0)
		{	
		}

	istream_iterator(istream_type& _Istr)
		: _Myistr(&_Istr)
		{	
		_Getval();
		}

	const _Ty& operator*() const
		{	
		return (_Myval);
		}

	const _Ty *operator->() const
		{	
		return (&**this);
		}

	_Myt& operator++()
		{	
		_Getval();
		return (*this);
		}

	_Myt operator++(int)
		{	
		_Myt _Tmp = *this;
		++*this;
		return (_Tmp);
		}

	bool _Equal(const _Myt& _Right) const
		{	
		return (_Myistr == _Right._Myistr);
		}

protected:
	void _Getval()
		{	
		if (_Myistr != 0 && !(*_Myistr >> _Myval))
			_Myistr = 0;
		}

	istream_type *_Myistr;	
	_Ty _Myval;	
	};

		
template<class _Ty, class _Elem, class _Traits, class _Diff> inline
	bool operator==(
		const istream_iterator<_Ty, _Elem, _Traits, _Diff>& _Left,
		const istream_iterator<_Ty, _Elem, _Traits, _Diff>& _Right)
	{	
	return (_Left._Equal(_Right));
	}

template<class _Ty, class _Elem, class _Traits, class _Diff> inline
	bool operator!=(
		const istream_iterator<_Ty, _Elem, _Traits, _Diff>& _Left,
		const istream_iterator<_Ty, _Elem, _Traits, _Diff>& _Right)
	{	
	return (!(_Left == _Right));
	}

		
template<class _Ty, class _Elem = char,
	class _Traits = char_traits<_Elem> >
	class ostream_iterator
		: public _Outit
	{	
public:
	typedef _Elem char_type;
	typedef _Traits traits_type;
	typedef basic_ostream<_Elem, _Traits> ostream_type;

	ostream_iterator(ostream_type& _Ostr,
		const _Elem *_Delim = 0)
		: _Myostr(&_Ostr), _Mydelim(_Delim)
		{	
		}

	ostream_iterator<_Ty, _Elem, _Traits>& operator=(const _Ty& _Val)
		{	
		*_Myostr << _Val;
		if (_Mydelim != 0)
			*_Myostr << _Mydelim;
		return (*this);
		}

	ostream_iterator<_Ty, _Elem, _Traits>& operator*()
		{	
		return (*this);
		}

	ostream_iterator<_Ty, _Elem, _Traits>& operator++()
		{	
		return (*this);
		}

	ostream_iterator<_Ty, _Elem, _Traits> operator++(int)
		{	
		return (*this);
		}

protected:
	const _Elem *_Mydelim;	
	ostream_type *_Myostr;	
	};

		
template<class _Iter> inline
	typename iterator_traits<_Iter>::value_type *_Val_type(_Iter)
	{	
	return (0);
	}


		
template<class _InIt, class _Diff> inline
	void advance(_InIt& _Where, _Diff _Off)
	{	
	_Advance(_Where, _Off, _Iter_cat(_Where));
	}

template<class _InIt, class _Diff> inline
	void _Advance(_InIt& _Where, _Diff _Off, input_iterator_tag)
	{	
	for (; 0 < _Off; --_Off)
		++_Where;
	}

template<class _FI, class _Diff> inline
	void _Advance(_FI& _Where, _Diff _Off, forward_iterator_tag)
	{	
	for (; 0 < _Off; --_Off)
		++_Where;
	}

template<class _BI, class _Diff> inline
	void _Advance(_BI& _Where, _Diff _Off, bidirectional_iterator_tag)
	{	
	for (; 0 < _Off; --_Off)
		++_Where;
	for (; _Off < 0; ++_Off)
		--_Where;
	}

template<class _RI, class _Diff> inline
	void _Advance(_RI& _Where, _Diff _Off, random_access_iterator_tag)
	{	
	_Where += _Off;
	}

		
template<class _Iter> inline
	typename iterator_traits<_Iter>::difference_type
		*_Dist_type(_Iter)
	{	
	return (0);
	}


}
#pragma warning(pop)
#pragma pack(pop)

#line 341 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\iterator"






















#line 55 "c:\\boost\\boost/detail/iterator.hpp"
















#line 72 "c:\\boost\\boost/detail/iterator.hpp"


    
namespace boost { namespace detail {


template <class Iterator>
struct iterator_traits
    : std::iterator_traits<Iterator>
{};
using std::distance;

}} 

















































































































































































































































































































































































































#line 489 "c:\\boost\\boost/detail/iterator.hpp"





#line 495 "c:\\boost\\boost/detail/iterator.hpp"
#line 33 "c:\\boost\\boost/array.hpp"
#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\algorithm"

#pragma once


#line 1 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\memory"

#pragma once





#pragma pack(push,8)
#pragma warning(push,3)
namespace std {

		
template<class _Ty> inline
	pair<_Ty  *, ptrdiff_t>
		get_temporary_buffer(ptrdiff_t _Count)
	{	
	_Ty  *_Pbuf;

	for (_Pbuf = 0; 0 < _Count; _Count /= 2)
		if ((_Pbuf = (_Ty  *)operator new(
			(size_t)_Count * sizeof (_Ty), nothrow)) != 0)
			break;

	return (pair<_Ty  *, ptrdiff_t>(_Pbuf, _Count));
	}

		
template<class _Ty> inline
	void return_temporary_buffer(_Ty *_Pbuf)
	{	
	operator delete(_Pbuf);
	}

		
template<class _InIt,
	class _FwdIt> inline
	_FwdIt _Uninit_copy(_InIt _First, _InIt _Last, _FwdIt _Dest,
		_Nonscalar_ptr_iterator_tag)
	{	
	_FwdIt _Next = _Dest;

	try {
	for (; _First != _Last; ++_Dest, ++_First)
		_Construct(&*_Dest, *_First);
	} catch (...) {
	for (; _Next != _Dest; ++_Next)
		_Destroy(&*_Next);
	throw;
	}
	return (_Dest);
	}

template<class _Ty1,
	class _Ty2> inline
	_Ty2 *_Uninit_copy(_Ty1 *_First, _Ty1 *_Last, _Ty2 *_Dest,
		_Scalar_ptr_iterator_tag)
	{	
	size_t _Count = (size_t)(_Last - _First);
	return ((_Ty2 *)memmove(&*_Dest, &*_First,
		_Count * sizeof (*_First)) + _Count);	
	}

template<class _InIt,
	class _FwdIt> inline
	_FwdIt uninitialized_copy(_InIt _First, _InIt _Last, _FwdIt _Dest)
	{	
	return (_Uninit_copy(_First, _Last, _Dest,
		_Ptr_cat(_First, _Dest)));
	}

		
template<class _InIt,
	class _FwdIt,
	class _Alloc> inline
	_FwdIt _Uninit_copy(_InIt _First, _InIt _Last, _FwdIt _Dest,
		_Alloc& _Al, _Nonscalar_ptr_iterator_tag)
	{	
	_FwdIt _Next = _Dest;

	try {
	for (; _First != _Last; ++_Dest, ++_First)
		_Al.construct(_Dest, *_First);
	} catch (...) {
	for (; _Next != _Dest; ++_Next)
		_Al.destroy(_Next);
	throw;
	}
	return (_Dest);
	}

template<class _InIt,
	class _FwdIt,
	class _Alloc> inline
	_FwdIt _Uninit_copy(_InIt _First, _InIt _Last, _FwdIt _Dest,
		_Alloc& _Al, _Scalar_ptr_iterator_tag)
	{	
	return (_Uninit_copy(_First, _Last, _Dest,
		_Al, _Nonscalar_ptr_iterator_tag()));
	}

template<class _Ty1,
	class _Ty2> inline
	_Ty2 *_Uninit_copy(_Ty1 *_First, _Ty1 *_Last, _Ty2 *_Dest,
		allocator<_Ty2>&, _Scalar_ptr_iterator_tag)
	{	
	size_t _Count = (size_t)(_Last - _First);
	return ((_Ty2 *)memmove(&*_Dest, &*_First,
		_Count * sizeof (*_First)) + _Count);	
	}

template<class _Ty1,
	class _Ty2> inline
	_Ty2 *_Uninit_copy(_Ty1 *_First, _Ty1 *_Last, _Ty2 *_Dest,
		allocator<const _Ty2>&, _Scalar_ptr_iterator_tag)
	{	
	size_t _Count = (size_t)(_Last - _First);
	return ((_Ty2 *)memmove(&*_Dest, &*_First,
		_Count * sizeof (*_First)) + _Count);	
	}

template<class _InIt,
	class _FwdIt,
	class _Alloc> inline
	_FwdIt _Uninitialized_copy(_InIt _First, _InIt _Last, _FwdIt _Dest,
		_Alloc& _Al)
	{	
	return (_Uninit_copy(_First, _Last, _Dest, _Al,
		_Ptr_cat(_First, _Dest)));
	}

		
template<class _FwdIt,
	class _Tval> inline
	void _Uninit_fill(_FwdIt _First, _FwdIt _Last, const _Tval& _Val,
		_Nonscalar_ptr_iterator_tag)
	{	
	_FwdIt _Next = _First;

	try {
	for (; _First != _Last; ++_First)
		_Construct(&*_First, _Val);
	} catch (...) {
	for (; _Next != _First; ++_Next)
		_Destroy(&*_Next);
	throw;
	}
	}

template<class _Ty,
	class _Tval> inline
	void _Uninit_fill(_Ty *_First, _Ty *_Last, const _Tval& _Val,
		_Scalar_ptr_iterator_tag)
	{	
	std::fill(_First, _Last, _Val);
	}

template<class _FwdIt,
	class _Tval> inline
	void uninitialized_fill(_FwdIt _First, _FwdIt _Last, const _Tval& _Val)
	{	
	_Uninit_fill(_First, _Last, _Val, _Ptr_cat(_First, _First));
	}

		
template<class _FwdIt,
	class _Diff,
	class _Tval> inline
	void _Uninit_fill_n(_FwdIt _First, _Diff _Count, const _Tval& _Val,
		_Nonscalar_ptr_iterator_tag)
	{	
	_FwdIt _Next = _First;

	try {
	for (; 0 < _Count; --_Count, ++_First)
		_Construct(&*_First, _Val);
	} catch (...) {
	for (; _Next != _First; ++_Next)
		_Destroy(&*_Next);
	throw;
	}
	}

template<class _Ty,
	class _Diff,
	class _Tval> inline
	void _Uninit_fill_n(_Ty *_First, _Diff _Count, const _Tval& _Val,
		_Scalar_ptr_iterator_tag)
	{	
	std::fill_n(_First, _Count, _Val);
	}

template<class _FwdIt,
	class _Diff,
	class _Tval> inline
	void uninitialized_fill_n(_FwdIt _First, _Diff _Count, const _Tval& _Val)
	{	
	_Uninit_fill_n(_First, _Count, _Val, _Ptr_cat(_First, _First));
	}

		
template<class _FwdIt,
	class _Diff,
	class _Tval,
	class _Alloc> inline
	void _Uninit_fill_n(_FwdIt _First, _Diff _Count,
		const _Tval& _Val, _Alloc& _Al, _Nonscalar_ptr_iterator_tag)
	{	
	_FwdIt _Next = _First;

	try {
	for (; 0 < _Count; --_Count, ++_First)
		_Al.construct(_First, _Val);
	} catch (...) {
	for (; _Next != _First; ++_Next)
		_Al.destroy(_Next);
	throw;
	}
	}

template<class _FwdIt,
	class _Diff,
	class _Tval,
	class _Alloc> inline
	void _Uninit_fill_n(_FwdIt _First, _Diff _Count,
		const _Tval& _Val, _Alloc& _Al, _Scalar_ptr_iterator_tag)
	{	
	_Uninit_fill_n(_First, _Count,
		_Val, _Al, _Nonscalar_ptr_iterator_tag());
	}

template<class _Ty,
	class _Diff,
	class _Tval> inline
	void _Uninit_fill_n(_Ty *_First, _Diff _Count,
		const _Tval& _Val, allocator<_Ty>&, _Scalar_ptr_iterator_tag)
	{	
	fill_n(_First, _Count, _Val);
	}

template<class _Ty,
	class _Diff,
	class _Tval> inline
	void _Uninit_fill_n(_Ty *_First, _Diff _Count,
		const _Tval& _Val, allocator<const _Ty>&, _Scalar_ptr_iterator_tag)
	{	
	fill_n(_First, _Count, _Val);
	}

template<class _FwdIt,
	class _Diff,
	class _Tval,
	class _Alloc> inline
	void _Uninitialized_fill_n(_FwdIt _First, _Diff _Count,
		const _Tval& _Val, _Alloc& _Al)
	{	
	_Uninit_fill_n(_First, _Count, _Val, _Al,
		_Ptr_cat(_First, _First));
	}

		
template<class _FwdIt,
	class _Ty>
	class raw_storage_iterator
		: public _Outit
	{	
public:
	typedef _FwdIt iterator_type;	
	typedef _FwdIt iter_type;	
	typedef _Ty element_type;	

	explicit raw_storage_iterator(_FwdIt _First)
		: _Next(_First)
		{	
		}

	raw_storage_iterator<_FwdIt, _Ty>& operator*()
		{	
		return (*this);
		}

	raw_storage_iterator<_FwdIt, _Ty>& operator=(const _Ty& _Val)
		{	
		_Construct(&*_Next, _Val);
		return (*this);
		}

	raw_storage_iterator<_FwdIt, _Ty>& operator++()
		{	
		++_Next;
		return (*this);
		}

	raw_storage_iterator<_FwdIt, _Ty> operator++(int)
		{	
		raw_storage_iterator<_FwdIt, _Ty> _Ans = *this;
		++_Next;
		return (_Ans);
		}

private:
	_FwdIt _Next;	
	};

		
template<class _Ty>
	class _Temp_iterator
		: public _Outit
	{	
public:
	typedef _Ty  *_Pty;

	_Temp_iterator(ptrdiff_t _Count = 0)
		{	
		pair<_Pty, ptrdiff_t> _Pair =
			std::get_temporary_buffer<_Ty>(_Count);
		_Buf._Begin = _Pair.first;
		_Buf._Current = _Pair.first;
		_Buf._Hiwater = _Pair.first;
		_Buf._Size = _Pair.second;
		_Pbuf = &_Buf;
		}

	_Temp_iterator(const _Temp_iterator<_Ty>& _Right)
		{	
		_Buf._Begin = 0;	
		_Buf._Current = 0;
		_Buf._Hiwater = 0;
		_Buf._Size = 0;
		*this = _Right;
		}

	~_Temp_iterator()
		{	
		if (_Buf._Begin != 0)
			{	
			for (_Pty _Next = _Buf._Begin;
				_Next != _Buf._Hiwater; ++_Next)
				_Destroy(&*_Next);
			std::return_temporary_buffer(_Buf._Begin);
			}
		}

	_Temp_iterator<_Ty>& operator=(const _Temp_iterator<_Ty>& _Right)
		{	
		_Pbuf = _Right._Pbuf;
		return (*this);
		}

	_Temp_iterator<_Ty>& operator=(const _Ty& _Val)
		{	
		if (_Pbuf->_Current < _Pbuf->_Hiwater)
			*_Pbuf->_Current++ = _Val;	
		else
			{	
			_Construct(&*_Pbuf->_Current, _Val);
			_Pbuf->_Hiwater = ++_Pbuf->_Current;
			}
		return (*this);
		}

	_Temp_iterator<_Ty>& operator*()
		{	
		return (*this);
		}

	_Temp_iterator<_Ty>& operator++()
		{	
		return (*this);
		}

	_Temp_iterator<_Ty>& operator++(int)
		{	
		return (*this);
		}

	_Temp_iterator<_Ty>& _Init()
		{	
		_Pbuf->_Current = _Pbuf->_Begin;
		return (*this);
		}

	_Pty _First() const
		{	
		return (_Pbuf->_Begin);
		}

	_Pty _Last() const
		{	
		return (_Pbuf->_Current);
		}

	ptrdiff_t _Maxlen() const
		{	
		return (_Pbuf->_Size);
		}

private:
	struct _Bufpar
		{	
		_Pty _Begin;	
		_Pty _Current;	
		_Pty _Hiwater;	
		ptrdiff_t _Size;	
		};
	_Bufpar _Buf;	
	_Bufpar *_Pbuf;	
	};

		
template<class _Ty>
	class auto_ptr;

template<class _Ty>
	struct auto_ptr_ref
		{	
	auto_ptr_ref(auto_ptr<_Ty>& _Right)
		: _Ref(_Right)
		{	
		}

	auto_ptr<_Ty>& _Ref;	
	};

template<class _Ty>
	class auto_ptr
		{	
public:
	typedef _Ty element_type;

	explicit auto_ptr(_Ty *_Ptr = 0) throw ()
		: _Myptr(_Ptr)
		{	
		}

	auto_ptr(auto_ptr<_Ty>& _Right) throw ()
		: _Myptr(_Right.release())
		{	
		}

	auto_ptr(auto_ptr_ref<_Ty> _Right) throw ()
		: _Myptr(_Right._Ref.release())
		{	
		}

	template<class _Other>
		operator auto_ptr<_Other>() throw ()
		{	
		return (auto_ptr<_Other>(*this));
		}

	template<class _Other>
		operator auto_ptr_ref<_Other>() throw ()
		{	
		return (auto_ptr_ref<_Other>(*this));
		}

	template<class _Other>
		auto_ptr<_Ty>& operator=(auto_ptr<_Other>& _Right) throw ()
		{	
		reset(_Right.release());
		return (*this);
		}

	template<class _Other>
		auto_ptr(auto_ptr<_Other>& _Right) throw ()
		: _Myptr(_Right.release())
		{	
		}

	auto_ptr<_Ty>& operator=(auto_ptr<_Ty>& _Right) throw ()
		{	
		reset(_Right.release());
		return (*this);
		}

	auto_ptr<_Ty>& operator=(auto_ptr_ref<_Ty>& _Right) throw ()
		{	
		reset(_Right._Ref.release());
		return (*this);
		}

	~auto_ptr()
		{	
		delete _Myptr;
		}

	_Ty& operator*() const throw ()
		{	
		return (*_Myptr);
		}

	_Ty *operator->() const throw ()
		{	
		return (&**this);
		}

	_Ty *get() const throw ()
		{	
		return (_Myptr);
		}

	_Ty *release() throw ()
		{	
		_Ty *_Tmp = _Myptr;
		_Myptr = 0;
		return (_Tmp);
		}

	void reset(_Ty* _Ptr = 0)
		{	
		if (_Ptr != _Myptr)
			delete _Myptr;
		_Myptr = _Ptr;
		}

private:
	_Ty *_Myptr;	
	};
}
#pragma warning(pop)
#pragma pack(pop)

#line 524 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\memory"






















#line 6 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\algorithm"

#pragma pack(push,8)
#pragma warning(push,3)
 #pragma warning(disable: 4244)
namespace std {

		
const int _ISORT_MAX = 32;	

		
template<class _InIt,
	class _Fn1> inline
	_Fn1 for_each(_InIt _First, _InIt _Last, _Fn1 _Func)
	{	
	for (; _First != _Last; ++_First)
		_Func(*_First);
	return (_Func);
	}

		
template<class _InIt,
	class _Ty> inline
	_InIt find(_InIt _First, _InIt _Last, const _Ty& _Val)
	{	
	for (; _First != _Last; ++_First)
		if (*_First == _Val)
			break;
	return (_First);
	}

inline const char *find(const char *_First, const char *_Last, int _Val)
	{	
	_First = (const char *)::memchr(_First, _Val, _Last - _First);
	return (_First == 0 ? _Last : _First);
	}

inline const signed char *find(const signed char *_First,
	const signed char *_Last, int _Val)
	{	
	_First = (const signed char *)::memchr(_First, _Val,
		_Last - _First);
	return (_First == 0 ? _Last : _First);
	}

inline const unsigned char *find(const unsigned char *_First,
	const unsigned char *_Last, int _Val)
	{	
	_First = (const unsigned char *)::memchr(_First, _Val,
		_Last - _First);
	return (_First == 0 ? _Last : _First);
	}

		
template<class _InIt,
	class _Pr> inline
	_InIt find_if(_InIt _First, _InIt _Last, _Pr _Pred)
	{	
	for (; _First != _Last; ++_First)
		if (_Pred(*_First))
			break;
	return (_First);
	}

		
template<class _FwdIt> inline
	_FwdIt adjacent_find(_FwdIt _First, _FwdIt _Last)
	{	
	for (_FwdIt _Firstb; (_Firstb = _First) != _Last && ++_First != _Last; )
		if (*_Firstb == *_First)
			return (_Firstb);
	return (_Last);
	}

		
template<class _FwdIt,
	class _Pr> inline
	_FwdIt adjacent_find(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	{	
	for (_FwdIt _Firstb; (_Firstb = _First) != _Last && ++_First != _Last; )
		if (_Pred(*_Firstb, *_First))
			return (_Firstb);
	return (_Last);
	}

		
template<class _InIt,
	class _Ty> inline
	typename iterator_traits<_InIt>::difference_type
		count(_InIt _First, _InIt _Last, const _Ty& _Val)
	{	
	typename iterator_traits<_InIt>::difference_type _Count = 0;

	for (; _First != _Last; ++_First)
		if (*_First == _Val)
			++_Count;
	return (_Count);
	}

		
template<class _InIt,
	class _Pr> inline
	typename iterator_traits<_InIt>::difference_type
		count_if(_InIt _First, _InIt _Last, _Pr _Pred)
	{	
	typename iterator_traits<_InIt>::difference_type _Count = 0;

	for (; _First != _Last; ++_First)
		if (_Pred(*_First))
			++_Count;
	return (_Count);
	}


		
template<class _FwdIt1,
	class _FwdIt2,
	class _Diff1,
	class _Diff2> inline
	_FwdIt1 _Search(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Diff1 *, _Diff2 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);
	_Diff2 _Count2 = 0;
	_Distance(_First2, _Last2, _Count2);

	for (; _Count2 <= _Count1; ++_First1, --_Count1)
		{	
		_FwdIt1 _Mid1 = _First1;
		for (_FwdIt2 _Mid2 = _First2; ; ++_Mid1, ++_Mid2)
			if (_Mid2 == _Last2)
				return (_First1);
			else if (!(*_Mid1 == *_Mid2))
				break;
		}
	return (_Last1);
	}

template<class _FwdIt1,
	class _FwdIt2> inline
	_FwdIt1 search(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2)
	{	
	return (_Search(_First1, _Last1, _First2, _Last2,
		_Dist_type(_First1), _Dist_type(_First2)));
	}

		
template<class _FwdIt1,
	class _FwdIt2,
	class _Diff1,
	class _Diff2,
	class _Pr> inline
	_FwdIt1 _Search(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Pr _Pred, _Diff1 *, _Diff2 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);
	_Diff2 _Count2 = 0;
	_Distance(_First2, _Last2, _Count2);

	for (; _Count2 <= _Count1; ++_First1, --_Count1)
		{	
		_FwdIt1 _Mid1 = _First1;
		for (_FwdIt2 _Mid2 = _First2; ; ++_Mid1, ++_Mid2)
			if (_Mid2 == _Last2)
				return (_First1);
			else if (!_Pred(*_Mid1, *_Mid2))
				break;
		}
	return (_Last1);
	}

template<class _FwdIt1,
	class _FwdIt2,
	class _Pr> inline
	_FwdIt1 search(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Pr _Pred)
	{	
	return (_Search(_First1, _Last1, _First2, _Last2, _Pred,
		_Dist_type(_First1), _Dist_type(_First2)));
	}

		
template<class _FwdIt1,
	class _Diff2,
	class _Ty,
	class _Diff1> inline
	_FwdIt1 _Search_n(_FwdIt1 _First1, _FwdIt1 _Last1,
		_Diff2 _Count, const _Ty& _Val, _Diff1 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);

	for (; _Count <= _Count1; ++_First1, --_Count1)
		{	
		_FwdIt1 _Mid1 = _First1;
		for (_Diff2 _Count2 = _Count; ; ++_Mid1, --_Count2)
			if (_Count2 == 0)
				return (_First1);
			else if (!(*_Mid1 == _Val))
				break;
		}
	return (_Last1);
	}

template<class _FwdIt1,
	class _Diff2,
	class _Ty> inline
	_FwdIt1 search_n(_FwdIt1 _First1, _FwdIt1 _Last1,
		_Diff2 _Count, const _Ty& _Val)
	{	
	return (_Search_n(_First1, _Last1, _Count, _Val, _Dist_type(_First1)));
	}

		
template<class _FwdIt1,
	class _Diff2,
	class _Ty,
	class _Diff1,
	class _Pr> inline
	_FwdIt1 _Search_n(_FwdIt1 _First1, _FwdIt1 _Last1,
		_Diff2 _Count, const _Ty& _Val, _Pr _Pred, _Diff1 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);

	for (; _Count <= _Count1; ++_First1, --_Count1)
		{	
		_FwdIt1 _Mid1 = _First1;
		for (_Diff2 _Count2 = _Count; ; ++_Mid1, --_Count2)
			if (_Count2 == 0)
				return (_First1);
			else if (!_Pred(*_Mid1, _Val))
				break;
		}
	return (_Last1);
	}

template<class _FwdIt1,
	class _Diff2,
	class _Ty,
	class _Pr> inline
	_FwdIt1 search_n(_FwdIt1 _First1, _FwdIt1 _Last1,
		_Diff2 _Count, const _Ty& _Val, _Pr _Pred)
	{	
	return (_Search_n(_First1, _Last1,
		_Count, _Val, _Pred, _Dist_type(_First1)));
	}

		
template<class _FwdIt1,
	class _FwdIt2,
	class _Diff1,
	class _Diff2> inline
	_FwdIt1 _Find_end(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Diff1 *, _Diff2 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);
	_Diff2 _Count2 = 0;
	_Distance(_First2, _Last2, _Count2);
	_FwdIt1 _Ans = _Last1;

	if (0 < _Count2)
		for (; _Count2 <= _Count1; ++_First1, --_Count1)
			{	
			_FwdIt1 _Mid1 = _First1;
			for (_FwdIt2 _Mid2 = _First2; ; ++_Mid1)
				if (!(*_Mid1 == *_Mid2))
					break;
				else if (++_Mid2 == _Last2)
					{	
					_Ans = _First1;
					break;
					}
			}
	return (_Ans);
	}

template<class _FwdIt1,
	class _FwdIt2> inline
	_FwdIt1 find_end(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2)
	{	
	return (_Find_end(_First1, _Last1, _First2, _Last2,
		_Dist_type(_First1), _Dist_type(_First2)));
	}

		
template<class _FwdIt1,
	class _FwdIt2,
	class _Diff1,
	class _Diff2,
	class _Pr> inline
	_FwdIt1 _Find_end(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Pr _Pred, _Diff1 *, _Diff2 *)
	{	
	_Diff1 _Count1 = 0;
	_Distance(_First1, _Last1, _Count1);
	_Diff2 _Count2 = 0;
	_Distance(_First2, _Last2, _Count2);
	_FwdIt1 _Ans = _Last1;

	if (0 < _Count2)
		for (; _Count2 <= _Count1; ++_First1, --_Count1)
			{	
			_FwdIt1 _Mid1 = _First1;
			for (_FwdIt2 _Mid2 = _First2; ; ++_Mid1)
				if (!_Pred(*_Mid1, *_Mid2))
					break;
				else if (++_Mid2 == _Last2)
					{	
					_Ans = _First1;
					break;
					}
			}
	return (_Ans);
	}

template<class _FwdIt1,
	class _FwdIt2,
	class _Pr> inline
	_FwdIt1 find_end(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Pr _Pred)
	{	
	return (_Find_end(_First1, _Last1, _First2, _Last2, _Pred,
		_Dist_type(_First1), _Dist_type(_First2)));
	}

		
template<class _FwdIt1,
	class _FwdIt2> inline
	_FwdIt1 find_first_of(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2)
	{	
	for (; _First1 != _Last1; ++_First1)
		for (_FwdIt2 _Mid2 = _First2; _Mid2 != _Last2; ++_Mid2)
			if (*_First1 == *_Mid2)
				return (_First1);
	return (_First1);
	}

		
template<class _FwdIt1,
	class _FwdIt2,
	class _Pr> inline
	_FwdIt1 find_first_of(_FwdIt1 _First1, _FwdIt1 _Last1,
		_FwdIt2 _First2, _FwdIt2 _Last2, _Pr _Pred)
	{	
	for (; _First1 != _Last1; ++_First1)
		for (_FwdIt2 _Mid2 = _First2; _Mid2 != _Last2; ++_Mid2)
			if (_Pred(*_First1, *_Mid2))
				return (_First1);
	return (_First1);
	}

		
template<class _FwdIt1,
	class _FwdIt2> inline
	void iter_swap(_FwdIt1 _Left, _FwdIt2 _Right)
	{	
	std::swap(*_Left, *_Right);
	}

		
template<class _FwdIt1,
	class _FwdIt2> inline
	_FwdIt2 swap_ranges(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2)
	{	
	for (; _First1 != _Last1; ++_First1, ++_First2)
		std::iter_swap(_First1, _First2);
	return (_First2);
	}

		
template<class _InIt,
	class _OutIt,
	class _Fn1> inline
	_OutIt transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func)
	{	
	for (; _First != _Last; ++_First, ++_Dest)
		*_Dest = _Func(*_First);
	return (_Dest);
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Fn2> inline
	_OutIt transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2,
		_OutIt _Dest, _Fn2 _Func)
	{	
	for (; _First1 != _Last1; ++_First1, ++_First2, ++_Dest)
		*_Dest = _Func(*_First1, *_First2);
	return (_Dest);
	}

		
template<class _FwdIt,
	class _Ty> inline
	void replace(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Oldval, const _Ty& _Newval)
	{	
	for (; _First != _Last; ++_First)
		if (*_First == _Oldval)
			*_First = _Newval;
	}

		
template<class _FwdIt,
	class _Pr,
	class _Ty> inline
	void replace_if(_FwdIt _First, _FwdIt _Last, _Pr _Pred, const _Ty& _Val)
	{	
	for (; _First != _Last; ++_First)
		if (_Pred(*_First))
			*_First = _Val;
	}

		
template<class _InIt,
	class _OutIt,
	class _Ty> inline
	_OutIt replace_copy(_InIt _First, _InIt _Last, _OutIt _Dest,
		const _Ty& _Oldval, const _Ty& _Newval)
	{	
	for (; _First != _Last; ++_First, ++_Dest)
		*_Dest = *_First == _Oldval ? _Newval : *_First;
	return (_Dest);
	}

		
template<class _InIt,
	class _OutIt,
	class _Pr,
	class _Ty> inline
	_OutIt replace_copy_if(_InIt _First, _InIt _Last, _OutIt _Dest,
		_Pr _Pred, const _Ty& _Val)
	{	
	for (; _First != _Last; ++_First, ++_Dest)
		*_Dest = _Pred(*_First) ? _Val : *_First;
	return (_Dest);
	}

		
template<class _FwdIt,
	class _Fn0> inline
	void generate(_FwdIt _First, _FwdIt _Last, _Fn0 _Func)
	{	
	for (; _First != _Last; ++_First)
		*_First = _Func();
	}

		
template<class _OutIt,
	class _Diff,
	class _Fn0> inline
	void generate_n(_OutIt _Dest, _Diff _Count, _Fn0 _Func)
	{	
	for (; 0 < _Count; --_Count, ++_Dest)
		*_Dest = _Func();
	}

		
template<class _InIt,
	class _OutIt,
	class _Ty> inline
	_OutIt remove_copy(_InIt _First, _InIt _Last,
		_OutIt _Dest, const _Ty& _Val)
	{	
	for (; _First != _Last; ++_First)
		if (!(*_First == _Val))
			*_Dest++ = *_First;
	return (_Dest);
	}

		
template<class _InIt,
	class _OutIt,
	class _Pr> inline
	_OutIt remove_copy_if(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First != _Last; ++_First)
		if (!_Pred(*_First))
			*_Dest++ = *_First;
	return (_Dest);
	}

		
template<class _FwdIt,
	class _Ty> inline
	_FwdIt remove(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	{	
	_First = find(_First, _Last, _Val);
	if (_First == _Last)
		return (_First);	
	else
		{	
		_FwdIt _First1 = _First;
		return (std::remove_copy(++_First1, _Last, _First, _Val));
		}
	}

		
template<class _FwdIt,
	class _Pr> inline
	_FwdIt remove_if(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	{	
	_First = std::find_if(_First, _Last, _Pred);
	if (_First == _Last)
		return (_First);	
	else
		{	
		_FwdIt _First1 = _First;
		return (std::remove_copy_if(++_First1, _Last, _First, _Pred));
		}
	}

		
template<class _FwdIt> inline
	_FwdIt unique(_FwdIt _First, _FwdIt _Last)
	{	
	for (_FwdIt _Firstb; (_Firstb = _First) != _Last && ++_First != _Last; )
		if (*_Firstb == *_First)
			{	
			for (; ++_First != _Last; )
				if (!(*_Firstb == *_First))
					*++_Firstb = *_First;
			return (++_Firstb);
			}
	return (_Last);
	}

		
template<class _FwdIt,
	class _Pr> inline
	_FwdIt unique(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	{	
	for (_FwdIt _Firstb; (_Firstb = _First) != _Last && ++_First != _Last; )
		if (_Pred(*_Firstb, *_First))
			{	
			for (; ++_First != _Last; )
				if (!_Pred(*_Firstb, *_First))
					*++_Firstb = *_First;
			return (++_Firstb);
			}
	return (_Last);
	}

		
template<class _InIt,
	class _OutIt,
	class _Ty> inline
	_OutIt _Unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest, _Ty *)
	{	
	_Ty _Val = *_First;

	for (*_Dest++ = _Val; ++_First != _Last; )
		if (!(_Val == *_First))
			_Val = *_First, *_Dest++ = _Val;
	return (_Dest);
	}

template<class _InIt,
	class _OutIt> inline
	_OutIt _Unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest,
		input_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, _Val_type(_First)));
	}

template<class _FwdIt,
	class _OutIt> inline
	_OutIt _Unique_copy(_FwdIt _First, _FwdIt _Last, _OutIt _Dest,
		forward_iterator_tag)
	{	
	_FwdIt _Firstb = _First;
	for (*_Dest++ = *_Firstb; ++_First != _Last; )
		if (!(*_Firstb == *_First))
			_Firstb = _First, *_Dest++ = *_Firstb;
	return (_Dest);
	}

template<class _BidIt,
	class _OutIt> inline
	_OutIt _Unique_copy(_BidIt _First, _BidIt _Last, _OutIt _Dest,
		bidirectional_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, forward_iterator_tag()));
	}

template<class _RanIt,
	class _OutIt> inline
	_OutIt _Unique_copy(_RanIt _First, _RanIt _Last, _OutIt _Dest,
		random_access_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, forward_iterator_tag()));
	}

template<class _InIt,
	class _OutIt> inline
	_OutIt unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest)
	{	
	return (_First == _Last ? _Dest :
		_Unique_copy(_First, _Last, _Dest, _Iter_cat(_First)));
	}

		
template<class _InIt,
	class _OutIt,
	class _Ty,
	class _Pr> inline
	_OutIt _Unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred,
		_Ty *)
	{	
	_Ty _Val = *_First;

	for (*_Dest++ = _Val; ++_First != _Last; )
		if (!_Pred(_Val, *_First))
			_Val = *_First, *_Dest++ = _Val;
	return (_Dest);
	}

template<class _InIt,
	class _OutIt,
	class _Pr> inline
	_OutIt _Unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred,
		input_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, _Pred, _Val_type(_First)));
	}

template<class _FwdIt,
	class _OutIt,
	class _Pr> inline
	_OutIt _Unique_copy(_FwdIt _First, _FwdIt _Last, _OutIt _Dest, _Pr _Pred,
		forward_iterator_tag)
	{	
	_FwdIt _Firstb = _First;

	for (*_Dest++ = *_Firstb; ++_First != _Last; )
		if (!_Pred(*_Firstb, *_First))
			_Firstb = _First, *_Dest++ = *_Firstb;
	return (_Dest);
	}

template<class _BidIt,
	class _OutIt,
	class _Pr> inline
	_OutIt _Unique_copy(_BidIt _First, _BidIt _Last, _OutIt _Dest, _Pr _Pred,
		bidirectional_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, _Pred,
		forward_iterator_tag()));
	}

template<class _RanIt,
	class _OutIt,
	class _Pr> inline
	_OutIt _Unique_copy(_RanIt _First, _RanIt _Last, _OutIt _Dest, _Pr _Pred,
		random_access_iterator_tag)
	{	
	return (_Unique_copy(_First, _Last, _Dest, _Pred,
		forward_iterator_tag()));
	}

template<class _InIt,
	class _OutIt,
	class _Pr> inline
	_OutIt unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred)
	{	
	return (_First == _Last ? _Dest
		: _Unique_copy(_First, _Last, _Dest, _Pred, _Iter_cat(_First)));
	}

		
template<class _BidIt> inline
	void _Reverse(_BidIt _First, _BidIt _Last, bidirectional_iterator_tag)
	{	
	for (; _First != _Last && _First != --_Last; ++_First)
		std::iter_swap(_First, _Last);
	}

template<class _RanIt> inline
	void _Reverse(_RanIt _First, _RanIt _Last, random_access_iterator_tag)
	{	
	for (; _First < _Last; ++_First)
		std::iter_swap(_First, --_Last);
	}

template<class _BidIt> inline
	void reverse(_BidIt _First, _BidIt _Last)
	{	
	_Reverse(_First, _Last, _Iter_cat(_First));
	}

		
template<class _BidIt,
	class _OutIt> inline
	_OutIt reverse_copy(_BidIt _First, _BidIt _Last, _OutIt _Dest)
	{	
	for (; _First != _Last; ++_Dest)
		*_Dest = *--_Last;
	return (_Dest);
	}

		
template<class _FwdIt> inline
	void _Rotate(_FwdIt _First, _FwdIt _Mid, _FwdIt _Last,
		forward_iterator_tag)
	{	
	for (_FwdIt _Next = _Mid; ; )
		{	
		std::iter_swap(_First, _Next);
		if (++_First == _Mid)
			if (++_Next == _Last)
				break;	
			else
				_Mid = _Next;	
		else if (++_Next == _Last)
			_Next = _Mid;	
		}
	}

template<class _BidIt> inline
	void _Rotate(_BidIt _First, _BidIt _Mid, _BidIt _Last,
		bidirectional_iterator_tag)
	{	
	std::reverse(_First, _Mid);
	std::reverse(_Mid, _Last);
	std::reverse(_First, _Last);
	}

template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Rotate(_RanIt _First, _RanIt _Mid, _RanIt _Last, _Diff *, _Ty *)
	{	
	_Diff _Shift = _Mid - _First;
	_Diff _Count = _Last - _First;

	for (_Diff _Factor = _Shift; _Factor != 0; )
		{	
		_Diff _Tmp = _Count % _Factor;
		_Count = _Factor, _Factor = _Tmp;
		}

	if (_Count < _Last - _First)
		for (; 0 < _Count; --_Count)
			{	
			_RanIt _Hole = _First + _Count;
			_RanIt _Next = _Hole;
			_Ty _Holeval = *_Hole;
			_RanIt _Next1 = _Next + _Shift == _Last ? _First : _Next + _Shift;
			while (_Next1 != _Hole)
				{	
				*_Next = *_Next1;
				_Next = _Next1;
				_Next1 = _Shift < _Last - _Next1 ? _Next1 + _Shift
					: _First + (_Shift - (_Last - _Next1));
				}
			*_Next = _Holeval;
			}
	}

template<class _RanIt> inline
	void _Rotate(_RanIt _First, _RanIt _Mid, _RanIt _Last,
			random_access_iterator_tag)
	{	
	_Rotate(_First, _Mid, _Last, _Dist_type(_First), _Val_type(_First));
	}

template<class _FwdIt> inline
	void rotate(_FwdIt _First, _FwdIt _Mid, _FwdIt _Last)
	{	
	if (_First != _Mid && _Mid != _Last)
		_Rotate(_First, _Mid, _Last, _Iter_cat(_First));
	}

		
template<class _FwdIt,
	class _OutIt> inline
	_OutIt rotate_copy(_FwdIt _First, _FwdIt _Mid, _FwdIt _Last, _OutIt _Dest)
	{	
	_Dest = std::copy(_Mid, _Last, _Dest);
	return (std::copy(_First, _Mid, _Dest));
	}

		
template<class _RanIt,
	class _Diff> inline
	void _Random_shuffle(_RanIt _First, _RanIt _Last, _Diff *)
	{	
	const int _RANDOM_BITS = 15;	
	const int _RANDOM_MAX = (1U << _RANDOM_BITS) - 1;

	_RanIt _Next = _First;
	for (unsigned long _Index = 2; ++_Next != _Last; ++_Index)
		{	
		unsigned long _Rm = _RANDOM_MAX;
		unsigned long _Rn = ::rand() & _RANDOM_MAX;
		for (; _Rm < _Index && _Rm != ~0UL;
			_Rm = _Rm << _RANDOM_BITS | _RANDOM_MAX)
			_Rn = _Rn << _RANDOM_BITS | _RANDOM_MAX;	

		std::iter_swap(_Next, _First + _Diff(_Rn % _Index));	
		}
	}

template<class _RanIt> inline
	void random_shuffle(_RanIt _First, _RanIt _Last)
	{	
	if (_First != _Last)
		_Random_shuffle(_First, _Last, _Dist_type(_First));
	}

		
template<class _RanIt,
	class _Fn1,
	class _Diff> inline
	void _Random_shuffle(_RanIt _First, _RanIt _Last, _Fn1& _Func, _Diff *)
	{	
	_RanIt _Next = _First;

	for (_Diff _Index = 2; ++_Next != _Last; ++_Index)
		std::iter_swap(_Next, _First + _Diff(_Func(_Index)));
	}

template<class _RanIt,
	class _Fn1> inline
	void random_shuffle(_RanIt _First, _RanIt _Last, _Fn1& _Func)
	{	
	if (_First != _Last)
		_Random_shuffle(_First, _Last, _Func, _Dist_type(_First));
	}

		
template<class _BidIt,
	class _Pr> inline
	_BidIt partition(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	for (; ; ++_First)
		{	
		for (; _First != _Last && _Pred(*_First); ++_First)
			;	
		if (_First == _Last)
			break;	

		for (; _First != --_Last && !_Pred(*_Last); )
			;	
		if (_First == _Last)
			break;	

		std::iter_swap(_First, _Last);	
		}
	return (_First);
	}

		
template<class _BidIt,
	class _Pr,
	class _Diff,
	class _Ty> inline
	_BidIt _Stable_partition(_BidIt _First, _BidIt _Last, _Pr _Pred,
		_Diff _Count, _Temp_iterator<_Ty>& _Tempbuf)
	{	
	if (_Count == 1)
		return (_Pred(*_First) ? _Last : _First);
	else if (_Count <= _Tempbuf._Maxlen())
		{	
		_BidIt _Next = _First;
		for (_Tempbuf._Init(); _First != _Last; ++_First)
			if (_Pred(*_First))
				*_Next++ = *_First;
			else
				*_Tempbuf++ = *_First;

		std::copy(_Tempbuf._First(), _Tempbuf._Last(), _Next);	
		return (_Next);
		}
	else
		{	
		_BidIt _Mid = _First;
		std::advance(_Mid, _Count / 2);

		_BidIt _Left = _Stable_partition(_First, _Mid, _Pred,
			_Count / 2, _Tempbuf);	
		_BidIt _Right = _Stable_partition(_Mid, _Last, _Pred,
			_Count - _Count / 2, _Tempbuf);	

		_Diff _Count1 = 0;
		_Distance(_Left, _Mid, _Count1);
		_Diff _Count2 = 0;
		_Distance(_Mid, _Right, _Count2);

		return (_Buffered_rotate(_Left, _Mid, _Right,
			_Count1, _Count2, _Tempbuf));	
		}
	}

template<class _BidIt,
	class _Pr,
	class _Diff,
	class _Ty> inline
	_BidIt _Stable_partition(_BidIt _First, _BidIt _Last, _Pr _Pred,
		_Diff *, _Ty *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	_Temp_iterator<_Ty> _Tempbuf(_Count);
	return (_Stable_partition(_First, _Last, _Pred, _Count, _Tempbuf));
	}

template<class _BidIt,
	class _Pr> inline
	_BidIt stable_partition(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	return (_First == _Last ? _First : _Stable_partition(_First, _Last, _Pred,
		_Dist_type(_First), _Val_type(_First)));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Push_heap(_RanIt _First, _Diff _Hole,
		_Diff _Top, _Ty _Val)
	{	
	for (_Diff _Idx = (_Hole - 1) / 2;
		_Top < _Hole && *(_First + _Idx) < _Val;
		_Idx = (_Hole - 1) / 2)
		{	
		*(_First + _Hole) = *(_First + _Idx);
		_Hole = _Idx;
		}

	*(_First + _Hole) = _Val;	
	}

template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Push_heap_0(_RanIt _First, _RanIt _Last, _Diff *, _Ty *)
	{	
	_Diff _Count = _Last - _First;
	if (0 < _Count)
		_Push_heap(_First, _Count, _Diff(0), _Ty(*_Last));
	}

template<class _RanIt> inline
	void push_heap(_RanIt _First, _RanIt _Last)
	{	
	if (_First != _Last)
		_Push_heap_0(_First, --_Last,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Push_heap(_RanIt _First, _Diff _Hole,
		_Diff _Top, _Ty _Val, _Pr _Pred)
	{	
	for (_Diff _Idx = (_Hole - 1) / 2;
		_Top < _Hole && _Pred(*(_First + _Idx), _Val);
		_Idx = (_Hole - 1) / 2)
		{	
		*(_First + _Hole) = *(_First + _Idx);
		_Hole = _Idx;
		}

	*(_First + _Hole) = _Val;	
	}

template<class _RanIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Push_heap_0(_RanIt _First, _RanIt _Last, _Pr _Pred, _Diff *, _Ty *)
	{	
	_Diff _Count = _Last - _First;
	if (0 < _Count)
		_Push_heap(_First, _Count, _Diff(0), _Ty(*_Last), _Pred);
	}

template<class _RanIt,
	class _Pr> inline
	void push_heap(_RanIt _First, _RanIt _Last, _Pr _Pred)
	{	
	if (_First != _Last)
		_Push_heap_0(_First, --_Last, _Pred,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Adjust_heap(_RanIt _First, _Diff _Hole, _Diff _Bottom, _Ty _Val)
	{	
	_Diff _Top = _Hole;
	_Diff _Idx = 2 * _Hole + 2;

	for (; _Idx < _Bottom; _Idx = 2 * _Idx + 2)
		{	
		if (*(_First + _Idx) < *(_First + (_Idx - 1)))
			--_Idx;
		*(_First + _Hole) = *(_First + _Idx), _Hole = _Idx;
		}

	if (_Idx == _Bottom)
		{	
		*(_First + _Hole) = *(_First + (_Bottom - 1));
		_Hole = _Bottom - 1;
		}
	_Push_heap(_First, _Hole, _Top, _Val);
	}

template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Pop_heap(_RanIt _First, _RanIt _Last, _RanIt _Dest,
		_Ty _Val, _Diff *)
	{	
	*_Dest = *_First;
	_Adjust_heap(_First, _Diff(0), _Diff(_Last - _First), _Val);
	}

template<class _RanIt,
	class _Ty> inline
	void _Pop_heap_0(_RanIt _First, _RanIt _Last, _Ty *)
	{	
	_Pop_heap(_First, _Last - 1, _Last - 1,
		_Ty(*(_Last - 1)), _Dist_type(_First));
	}

template<class _RanIt> inline
	void pop_heap(_RanIt _First, _RanIt _Last)
	{	
	if (1 < _Last - _First)
		_Pop_heap_0(_First, _Last, _Val_type(_First));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Adjust_heap(_RanIt _First, _Diff _Hole, _Diff _Bottom,
		_Ty _Val, _Pr _Pred)
	{	
	_Diff _Top = _Hole;
	_Diff _Idx = 2 * _Hole + 2;

	for (; _Idx < _Bottom; _Idx = 2 * _Idx + 2)
		{	
		if (_Pred(*(_First + _Idx), *(_First + (_Idx - 1))))
			--_Idx;
		*(_First + _Hole) = *(_First + _Idx), _Hole = _Idx;
		}

	if (_Idx == _Bottom)
		{	
		*(_First + _Hole) = *(_First + (_Bottom - 1));
		_Hole = _Bottom - 1;
		}
	_Push_heap(_First, _Hole, _Top, _Val, _Pred);
	}

template<class _RanIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Pop_heap(_RanIt _First, _RanIt _Last, _RanIt _Dest,
		_Ty _Val, _Pr _Pred, _Diff *)
	{	
	*_Dest = *_First;
	_Adjust_heap(_First, _Diff(0), _Diff(_Last - _First), _Val, _Pred);
	}

template<class _RanIt,
	class _Ty,
	class _Pr> inline
	void _Pop_heap_0(_RanIt _First, _RanIt _Last, _Pr _Pred, _Ty *)
	{	
	_Pop_heap(_First, _Last - 1, _Last - 1,
		_Ty(*(_Last - 1)), _Pred, _Dist_type(_First));
	}

template<class _RanIt,
	class _Pr> inline
	void pop_heap(_RanIt _First, _RanIt _Last, _Pr _Pred)
	{	
	if (1 < _Last - _First)
		_Pop_heap_0(_First, _Last, _Pred, _Val_type(_First));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty> inline
	void _Make_heap(_RanIt _First, _RanIt _Last, _Diff *, _Ty *)
	{	
	_Diff _Bottom = _Last - _First;

	for (_Diff _Hole = _Bottom / 2; 0 < _Hole; )
		{	
		--_Hole;
		_Adjust_heap(_First, _Hole, _Bottom, _Ty(*(_First + _Hole)));
		}
	}

template<class _RanIt> inline
	void make_heap(_RanIt _First, _RanIt _Last)
	{	
	if (1 < _Last - _First)
		_Make_heap(_First, _Last,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _RanIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Make_heap(_RanIt _First, _RanIt _Last, _Pr _Pred, _Diff *, _Ty *)
	{	
	_Diff _Bottom = _Last - _First;
	for (_Diff _Hole = _Bottom / 2; 0 < _Hole; )
		{	
		--_Hole;
		_Adjust_heap(_First, _Hole, _Bottom,
			_Ty(*(_First + _Hole)), _Pred);
		}
	}

template<class _RanIt,
	class _Pr> inline
	void make_heap(_RanIt _First, _RanIt _Last, _Pr _Pred)
	{	
	if (1 < _Last - _First)
		_Make_heap(_First, _Last, _Pred,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _RanIt> inline
	void sort_heap(_RanIt _First, _RanIt _Last)
	{	
	for (; 1 < _Last - _First; --_Last)
		std::pop_heap(_First, _Last);
	}

		
template<class _RanIt,
	class _Pr> inline
	void sort_heap(_RanIt _First, _RanIt _Last, _Pr _Pred)
	{	
	for (; 1 < _Last - _First; --_Last)
		std::pop_heap(_First, _Last, _Pred);
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff> inline
	_FwdIt _Lower_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);

	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (*_Mid < _Val)
			_First = ++_Mid, _Count -= _Count2 + 1;
		else
			_Count = _Count2;
		}
	return (_First);
	}

template<class _FwdIt,
	class _Ty> inline
	_FwdIt lower_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	{	
	return (_Lower_bound(_First, _Last, _Val, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff,
	class _Pr> inline
	_FwdIt _Lower_bound(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (_Pred(*_Mid, _Val))
			_First = ++_Mid, _Count -= _Count2 + 1;
		else
			_Count = _Count2;
		}
	return (_First);
	}

template<class _FwdIt,
	class _Ty,
	class _Pr> inline
	_FwdIt lower_bound(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred)
	{	
	return (_Lower_bound(_First, _Last, _Val, _Pred, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff> inline
	_FwdIt _Upper_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (!(_Val < *_Mid))
			_First = ++_Mid, _Count -= _Count2 + 1;
		else
			_Count = _Count2;
		}
	return (_First);
	}

template<class _FwdIt,
	class _Ty> inline
	_FwdIt upper_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	{	
	return (_Upper_bound(_First, _Last, _Val, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff,
	class _Pr> inline
	_FwdIt _Upper_bound(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (!_Pred(_Val, *_Mid))
			_First = ++_Mid, _Count -= _Count2 + 1;
		else
			_Count = _Count2;
		}
	return (_First);
	}

template<class _FwdIt,
	class _Ty,
	class _Pr> inline
	_FwdIt upper_bound(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred)
	{	
	return (_Upper_bound(_First, _Last, _Val, _Pred, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff> inline
	pair<_FwdIt, _FwdIt> _Equal_range(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);

	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (*_Mid < _Val)
			{	
			_First = ++_Mid;
			_Count -= _Count2 + 1;
			}
		else if (_Val < *_Mid)
			_Count = _Count2;	
		else
			{	
			_FwdIt _First2 = lower_bound(_First, _Mid, _Val);
			std::advance(_First, _Count);
			_FwdIt _Last2 = upper_bound(++_Mid, _First, _Val);
			return (pair<_FwdIt, _FwdIt>(_First2, _Last2));
			}
		}

	return (pair<_FwdIt, _FwdIt>(_First, _First));	
	}

template<class _FwdIt,
	class _Ty> inline
	pair<_FwdIt, _FwdIt> equal_range(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val)
	{	
	return (_Equal_range(_First, _Last, _Val, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Diff,
	class _Pr> inline
	pair<_FwdIt, _FwdIt> _Equal_range(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred, _Diff *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);

	for (; 0 < _Count; )
		{	
		_Diff _Count2 = _Count / 2;
		_FwdIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (_Pred(*_Mid, _Val))
			{	
			_First = ++_Mid;
			_Count -= _Count2 + 1;
			}
		else if (_Pred(_Val, *_Mid))
			_Count = _Count2;	
		else
			{	
			_FwdIt _First2 = lower_bound(_First, _Mid, _Val, _Pred);
			std::advance(_First, _Count);
			_FwdIt _Last2 = upper_bound(++_Mid, _First, _Val, _Pred);
			return (pair<_FwdIt, _FwdIt>(_First2, _Last2));
			}
		}

	return (pair<_FwdIt, _FwdIt>(_First, _First));	
	}

template<class _FwdIt,
	class _Ty,
	class _Pr> inline
	pair<_FwdIt, _FwdIt> equal_range(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred)
	{	
	return (_Equal_range(_First, _Last, _Val, _Pred, _Dist_type(_First)));
	}

		
template<class _FwdIt,
	class _Ty> inline
	bool binary_search(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	{	
	_First = std::lower_bound(_First, _Last, _Val);
	return (_First != _Last && !(_Val < *_First));
	}

		
template<class _FwdIt,
	class _Ty,
	class _Pr> inline
	bool binary_search(_FwdIt _First, _FwdIt _Last,
		const _Ty& _Val, _Pr _Pred)
	{	
	_First = std::lower_bound(_First, _Last, _Val, _Pred);
	return (_First != _Last && !_Pred(_Val, *_First));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt> inline
	_OutIt merge(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; ++_Dest)
		if (*_First2 < *_First1)
			*_Dest = *_First2, ++_First2;
		else
			*_Dest = *_First1, ++_First1;

	_Dest = std::copy(_First1, _Last1, _Dest);	
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Pr> inline
	_OutIt merge(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; ++_Dest)
		if (_Pred(*_First2, *_First1))
			*_Dest = *_First2, ++_First2;
		else
			*_Dest = *_First1, ++_First1;

	_Dest = std::copy(_First1, _Last1, _Dest);	
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _BidIt,
	class _Diff,
	class _Ty> inline
	_BidIt _Buffered_rotate(_BidIt _First, _BidIt _Mid, _BidIt _Last,
		_Diff _Count1, _Diff _Count2, _Temp_iterator<_Ty>& _Tempbuf)
	{	
	if (_Count1 <= _Count2 && _Count1 <= _Tempbuf._Maxlen())
		{	
		std::copy(_First, _Mid, _Tempbuf._Init());
		std::copy(_Mid, _Last, _First);
		return (std::copy_backward(_Tempbuf._First(), _Tempbuf._Last(),
			_Last));
		}
	else if (_Count2 <= _Tempbuf._Maxlen())
		{	
		std::copy(_Mid, _Last, _Tempbuf._Init());
		std::copy_backward(_First, _Mid, _Last);
		return (std::copy(_Tempbuf._First(), _Tempbuf._Last(), _First));
		}
	else
		{	
		std::rotate(_First, _Mid, _Last);
		std::advance(_First, _Count2);
		return (_First);
		}
	}

template<class _BidIt1,
	class _BidIt2,
	class _BidIt3> inline
	_BidIt3 _Merge_backward(_BidIt1 _First1, _BidIt1 _Last1,
		_BidIt2 _First2, _BidIt2 _Last2, _BidIt3 _Dest)
	{	
	for (; ; )
		if (_First1 == _Last1)
			return (std::copy_backward(_First2, _Last2, _Dest));
		else if (_First2 == _Last2)
			return (std::copy_backward(_First1, _Last1, _Dest));
		else if (*--_Last2 < *--_Last1)
			*--_Dest = *_Last1, ++_Last2;
		else
			*--_Dest = *_Last2, ++_Last1;
	}

template<class _BidIt,
	class _Diff,
	class _Ty> inline
	void _Buffered_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last,
		_Diff _Count1, _Diff _Count2,
			_Temp_iterator<_Ty>& _Tempbuf)
	{	
	if (_Count1 + _Count2 == 2)
		{	
		if (*_Mid < *_First)
			std::iter_swap(_First, _Mid);
		}
	else if (_Count1 <= _Count2 && _Count1 <= _Tempbuf._Maxlen())
		{	
		std::copy(_First, _Mid, _Tempbuf._Init());
		std::merge(_Tempbuf._First(), _Tempbuf._Last(), _Mid, _Last, _First);
		}
	else if (_Count2 <= _Tempbuf._Maxlen())
		{	
		std::copy(_Mid, _Last, _Tempbuf._Init());
		_Merge_backward(_First, _Mid,
			_Tempbuf._First(), _Tempbuf._Last(), _Last);
		}
	else
		{	
		_BidIt _Firstn, _Lastn;
		_Diff _Count1n, _Count2n;

		if (_Count2 < _Count1)
			{	
			_Count1n = _Count1 / 2, _Count2n = 0;
			_Firstn = _First;
			std::advance(_Firstn, _Count1n);
			_Lastn = std::lower_bound(_Mid, _Last, *_Firstn);
			_Distance(_Mid, _Lastn, _Count2n);
			}
		else
			{	
			_Count1n = 0, _Count2n = _Count2 / 2;
			_Lastn = _Mid;
			std::advance(_Lastn, _Count2n);
			_Firstn = std::upper_bound(_First, _Mid, *_Lastn);
			_Distance(_First, _Firstn, _Count1n);
			}

		_BidIt _Midn = _Buffered_rotate(_Firstn, _Mid, _Lastn,
			_Count1 - _Count1n, _Count2n, _Tempbuf);	
		_Buffered_merge(_First, _Firstn, _Midn,
			_Count1n, _Count2n, _Tempbuf);	
		_Buffered_merge(_Midn, _Lastn, _Last,
			_Count1 - _Count1n, _Count2 - _Count2n, _Tempbuf);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty> inline
	void _Inplace_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last,
		_Diff *, _Ty *)
	{	
	_Diff _Count1 = 0;
	_Distance(_First, _Mid, _Count1);
	_Diff _Count2 = 0;
	_Distance(_Mid, _Last, _Count2);
	_Temp_iterator<_Ty> _Tempbuf(_Count1 < _Count2 ? _Count1 : _Count2);
	_Buffered_merge(_First, _Mid, _Last,
		_Count1, _Count2, _Tempbuf);
	}

template<class _BidIt> inline
	void inplace_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last)
	{	
	if (_First != _Mid && _Mid != _Last)
		_Inplace_merge(_First, _Mid, _Last,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _BidIt1,
	class _BidIt2,
	class _BidIt3,
	class _Pr> inline
	_BidIt3 _Merge_backward(_BidIt1 _First1, _BidIt1 _Last1,
		_BidIt2 _First2, _BidIt2 _Last2, _BidIt3 _Dest, _Pr _Pred)
	{	
	for (; ; )
		if (_First1 == _Last1)
			return (std::copy_backward(_First2, _Last2, _Dest));
		else if (_First2 == _Last2)
			return (std::copy_backward(_First1, _Last1, _Dest));
		else if (_Pred(*--_Last2, *--_Last1))
			*--_Dest = *_Last1, ++_Last2;
		else
			*--_Dest = *_Last2, ++_Last1;
	}

template<class _BidIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Buffered_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last,
		_Diff _Count1, _Diff _Count2,
			_Temp_iterator<_Ty>& _Tempbuf, _Pr _Pred)
	{	
	if (_Count1 + _Count2 == 2)
		{	
		if (_Pred(*_Mid, *_First))
			std::iter_swap(_First, _Mid);
		}
	else if (_Count1 <= _Count2 && _Count1 <= _Tempbuf._Maxlen())
		{	
		std::copy(_First, _Mid, _Tempbuf._Init());
		std::merge(_Tempbuf._First(), _Tempbuf._Last(),
			_Mid, _Last, _First, _Pred);
		}
	else if (_Count2 <= _Tempbuf._Maxlen())
		{	
		std::copy(_Mid, _Last, _Tempbuf._Init());
		_Merge_backward(_First, _Mid, _Tempbuf._First(), _Tempbuf._Last(),
			_Last, _Pred);
		}
	else
		{	
		_BidIt _Firstn, _Lastn;
		_Diff _Count1n, _Count2n;
		if (_Count2 < _Count1)
			{	
			_Count1n = _Count1 / 2, _Count2n = 0;
			_Firstn = _First;
			std::advance(_Firstn, _Count1n);
			_Lastn = lower_bound(_Mid, _Last, *_Firstn, _Pred);
			_Distance(_Mid, _Lastn, _Count2n);
			}
		else
			{	
			_Count1n = 0, _Count2n = _Count2 / 2;
			_Lastn = _Mid;
			std::advance(_Lastn, _Count2n);
			_Firstn = upper_bound(_First, _Mid, *_Lastn, _Pred);
			_Distance(_First, _Firstn, _Count1n);
			}
		_BidIt _Midn = _Buffered_rotate(_Firstn, _Mid, _Lastn,
			_Count1 - _Count1n, _Count2n, _Tempbuf);	
		_Buffered_merge(_First, _Firstn, _Midn,
			_Count1n, _Count2n, _Tempbuf, _Pred);	
		_Buffered_merge(_Midn, _Lastn, _Last,
			_Count1 - _Count1n, _Count2 - _Count2n, _Tempbuf, _Pred);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Inplace_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last, _Pr _Pred,
		_Diff *, _Ty *)
	{	
	_Diff _Count1 = 0;
	_Distance(_First, _Mid, _Count1);
	_Diff _Count2 = 0;
	_Distance(_Mid, _Last, _Count2);
	_Temp_iterator<_Ty> _Tempbuf(_Count1 < _Count2 ? _Count1 : _Count2);
	_Buffered_merge(_First, _Mid, _Last,
		_Count1, _Count2, _Tempbuf, _Pred);
	}

template<class _BidIt,
	class _Pr> inline
	void inplace_merge(_BidIt _First, _BidIt _Mid, _BidIt _Last, _Pr _Pred)
	{	
	if (_First != _Mid && _Mid != _Last)
		_Inplace_merge(_First, _Mid, _Last, _Pred,
			_Dist_type(_First), _Val_type(_First));
	}

		
template<class _BidIt> inline
	void _Insertion_sort(_BidIt _First, _BidIt _Last)
	{	
	if (_First != _Last)
		for (_BidIt _Next = _First; ++_Next != _Last; )
			if (*_Next < *_First)
				{	
				_BidIt _Next1 = _Next;
				std::rotate(_First, _Next, ++_Next1);
				}
			else
				{	
				_BidIt _Dest = _Next;
				for (_BidIt _Dest0 = _Dest; *_Next < *--_Dest0; )
					_Dest = _Dest0;
				if (_Dest != _Next)
					{	
					_BidIt _Next1 = _Next;
					std::rotate(_Dest, _Next, ++_Next1);
					}
				}
	}

template<class _RanIt> inline
	void _Med3(_RanIt _First, _RanIt _Mid, _RanIt _Last)
	{	
	if (*_Mid < *_First)
		std::iter_swap(_Mid, _First);
	if (*_Last < *_Mid)
		std::iter_swap(_Last, _Mid);
	if (*_Mid < *_First)
		std::iter_swap(_Mid, _First);
	}

template<class _RanIt> inline
	void _Median(_RanIt _First, _RanIt _Mid, _RanIt _Last)
	{	
	if (40 < _Last - _First)
		{	
		int _Step = (_Last - _First + 1) / 8;
		_Med3(_First, _First + _Step, _First + 2 * _Step);
		_Med3(_Mid - _Step, _Mid, _Mid + _Step);
		_Med3(_Last - 2 * _Step, _Last - _Step, _Last);
		_Med3(_First + _Step, _Mid, _Last - _Step);
		}
	else
		_Med3(_First, _Mid, _Last);
	}

template<class _RanIt> inline
	pair<_RanIt, _RanIt> _Unguarded_partition(_RanIt _First, _RanIt _Last)
	{	
	_RanIt _Mid = _First + (_Last - _First) / 2;	
	_Median(_First, _Mid, _Last - 1);
	_RanIt _Pfirst = _Mid;
	_RanIt _Plast = _Pfirst + 1;

	while (_First < _Pfirst
		&& !(*(_Pfirst - 1) < *_Pfirst)
		&& !(*_Pfirst < *(_Pfirst - 1)))
		--_Pfirst;
	while (_Plast < _Last
		&& !(*_Plast < *_Pfirst)
		&& !(*_Pfirst < *_Plast))
		++_Plast;

	_RanIt _Gfirst = _Plast;
	_RanIt _Glast = _Pfirst;

	for (; ; )
		{	
		for (; _Gfirst < _Last; ++_Gfirst)
			if (*_Pfirst < *_Gfirst)
				;
			else if (*_Gfirst < *_Pfirst)
				break;
			else
				std::iter_swap(_Plast++, _Gfirst);
		for (; _First < _Glast; --_Glast)
			if (*(_Glast - 1) < *_Pfirst)
				;
			else if (*_Pfirst < *(_Glast - 1))
				break;
			else
				std::iter_swap(--_Pfirst, _Glast - 1);
		if (_Glast == _First && _Gfirst == _Last)
			return (pair<_RanIt, _RanIt>(_Pfirst, _Plast));

		if (_Glast == _First)
			{	
			if (_Plast != _Gfirst)
				std::iter_swap(_Pfirst, _Plast);
			++_Plast;
			std::iter_swap(_Pfirst++, _Gfirst++);
			}
		else if (_Gfirst == _Last)
			{	
			if (--_Glast != --_Pfirst)
				std::iter_swap(_Glast, _Pfirst);
			std::iter_swap(_Pfirst, --_Plast);
			}
		else
			std::iter_swap(_Gfirst++, --_Glast);
		}
	}

template<class _RanIt,
	class _Diff> inline
	void _Sort(_RanIt _First, _RanIt _Last, _Diff _Ideal)
	{	
	_Diff _Count;
	for (; _ISORT_MAX < (_Count = _Last - _First) && 0 < _Ideal; )
		{	
		pair<_RanIt, _RanIt> _Mid = _Unguarded_partition(_First, _Last);
		_Ideal /= 2, _Ideal += _Ideal / 2;	

		if (_Mid.first - _First < _Last - _Mid.second)	
			_Sort(_First, _Mid.first, _Ideal), _First = _Mid.second;
		else
			_Sort(_Mid.second, _Last, _Ideal), _Last = _Mid.first;
		}

	if (_ISORT_MAX < _Count)
		{	
		std::make_heap(_First, _Last);
		std::sort_heap(_First, _Last);
		}
	else if (1 < _Count)
		_Insertion_sort(_First, _Last);	
	}

template<class _RanIt> inline
	void sort(_RanIt _First, _RanIt _Last)
	{	
	_Sort(_First, _Last, _Last - _First);
	}

		
template<class _BidIt,
	class _Pr> inline
	void _Insertion_sort(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	if (_First != _Last)
		for (_BidIt _Next = _First; ++_Next != _Last; )
			if (_Pred(*_Next, *_First))
				{	
				_BidIt _Next1 = _Next;
				std::rotate(_First, _Next, ++_Next1);
				}
			else
				{	
				_BidIt _Dest = _Next;
				for (_BidIt _Dest0 = _Dest; _Pred(*_Next, *--_Dest0); )
					_Dest = _Dest0;
				if (_Dest != _Next)
					{	
					_BidIt _Next1 = _Next;
					std::rotate(_Dest, _Next, ++_Next1);
					}
				}
	}

template<class _RanIt,
	class _Pr> inline
	void _Med3(_RanIt _First, _RanIt _Mid, _RanIt _Last, _Pr _Pred)
	{	
	if (_Pred(*_Mid, *_First))
		std::iter_swap(_Mid, _First);
	if (_Pred(*_Last, *_Mid))
		std::iter_swap(_Last, _Mid);
	if (_Pred(*_Mid, *_First))
		std::iter_swap(_Mid, _First);
	}

template<class _RanIt,
	class _Pr> inline
	void _Median(_RanIt _First, _RanIt _Mid, _RanIt _Last, _Pr _Pred)
	{	
	if (40 < _Last - _First)
		{	
		int _Step = (_Last - _First + 1) / 8;
		_Med3(_First, _First + _Step, _First + 2 * _Step, _Pred);
		_Med3(_Mid - _Step, _Mid, _Mid + _Step, _Pred);
		_Med3(_Last - 2 * _Step, _Last - _Step, _Last, _Pred);
		_Med3(_First + _Step, _Mid, _Last - _Step, _Pred);
		}
	else
		_Med3(_First, _Mid, _Last, _Pred);
	}

template<class _RanIt,
	class _Pr> inline
	pair<_RanIt, _RanIt> _Unguarded_partition(_RanIt _First, _RanIt _Last,
		_Pr _Pred)
	{	
	_RanIt _Mid = _First + (_Last - _First) / 2;
	_Median(_First, _Mid, _Last - 1, _Pred);
	_RanIt _Pfirst = _Mid;
	_RanIt _Plast = _Pfirst + 1;

	while (_First < _Pfirst
		&& !_Pred(*(_Pfirst - 1), *_Pfirst)
		&& !_Pred(*_Pfirst, *(_Pfirst - 1)))
		--_Pfirst;
	while (_Plast < _Last
		&& !_Pred(*_Plast, *_Pfirst)
		&& !_Pred(*_Pfirst, *_Plast))
		++_Plast;

	_RanIt _Gfirst = _Plast;
	_RanIt _Glast = _Pfirst;

	for (; ; )
		{	
		for (; _Gfirst < _Last; ++_Gfirst)
			if (_Pred(*_Pfirst, *_Gfirst))
				;
			else if (_Pred(*_Gfirst, *_Pfirst))
				break;
			else
				std::iter_swap(_Plast++, _Gfirst);
		for (; _First < _Glast; --_Glast)
			if (_Pred(*(_Glast - 1), *_Pfirst))
				;
			else if (_Pred(*_Pfirst, *(_Glast - 1)))
				break;
			else
				std::iter_swap(--_Pfirst, _Glast - 1);
		if (_Glast == _First && _Gfirst == _Last)
			return (pair<_RanIt, _RanIt>(_Pfirst, _Plast));

		if (_Glast == _First)
			{	
			if (_Plast != _Gfirst)
				std::iter_swap(_Pfirst, _Plast);
			++_Plast;
			std::iter_swap(_Pfirst++, _Gfirst++);
			}
		else if (_Gfirst == _Last)
			{	
			if (--_Glast != --_Pfirst)
				std::iter_swap(_Glast, _Pfirst);
			std::iter_swap(_Pfirst, --_Plast);
			}
		else
			std::iter_swap(_Gfirst++, --_Glast);
		}
	}

template<class _RanIt,
	class _Diff,
	class _Pr> inline
	void _Sort(_RanIt _First, _RanIt _Last, _Diff _Ideal, _Pr _Pred)
	{	
	_Diff _Count;
	for (; _ISORT_MAX < (_Count = _Last - _First) && 0 < _Ideal; )
		{	
		pair<_RanIt, _RanIt> _Mid =
			_Unguarded_partition(_First, _Last, _Pred);
		_Ideal /= 2, _Ideal += _Ideal / 2;	

		if (_Mid.first - _First < _Last - _Mid.second)	
			_Sort(_First, _Mid.first, _Ideal, _Pred), _First = _Mid.second;
		else
			_Sort(_Mid.second, _Last, _Ideal, _Pred), _Last = _Mid.first;
		}

	if (_ISORT_MAX < _Count)
		{	
		std::make_heap(_First, _Last, _Pred);
		std::sort_heap(_First, _Last, _Pred);
		}
	else if (1 < _Count)
		_Insertion_sort(_First, _Last, _Pred);	
	}

template<class _RanIt,
	class _Pr> inline
	void sort(_RanIt _First, _RanIt _Last, _Pr _Pred)
	{	
	_Sort(_First, _Last, _Last - _First, _Pred);
	}

		
template<class _BidIt,
	class _OutIt,
	class _Diff> inline
	void _Chunked_merge(_BidIt _First, _BidIt _Last, _OutIt _Dest,
		_Diff _Chunk, _Diff _Count)
	{	
	for (_Diff _Chunk2 = _Chunk * 2; _Chunk2 <= _Count; _Count -= _Chunk2)
		{	
		_BidIt _Mid1 = _First;
		std::advance(_Mid1, _Chunk);
		_BidIt _Mid2 = _Mid1;
		std::advance(_Mid2, _Chunk);

		_Dest = std::merge(_First, _Mid1, _Mid1, _Mid2, _Dest);
		_First = _Mid2;
		}

	if (_Count <= _Chunk)
		std::copy(_First, _Last, _Dest);	
	else
		{	
		_BidIt _Mid = _First;
		std::advance(_Mid, _Chunk);

		std::merge(_First, _Mid, _Mid, _Last, _Dest);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty> inline
	void _Buffered_merge_sort(_BidIt _First, _BidIt _Last, _Diff _Count,
		_Temp_iterator<_Ty>& _Tempbuf)
	{	
	_BidIt _Mid = _First;
	for (_Diff _Nleft = _Count; _ISORT_MAX <= _Nleft; _Nleft -= _ISORT_MAX)
		{	
		_BidIt _Midend = _Mid;
		std::advance(_Midend, (int)_ISORT_MAX);

		_Insertion_sort(_Mid, _Midend);
		_Mid = _Midend;
		}
	_Insertion_sort(_Mid, _Last);	

	for (_Diff _Chunk = _ISORT_MAX; _Chunk < _Count; _Chunk *= 2)
		{	
		_Chunked_merge(_First, _Last, _Tempbuf._Init(),
			_Chunk, _Count);
		_Chunked_merge(_Tempbuf._First(), _Tempbuf._Last(), _First,
			_Chunk *= 2, _Count);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty> inline
	void _Stable_sort(_BidIt _First, _BidIt _Last, _Diff _Count,
		_Temp_iterator<_Ty>& _Tempbuf)
	{	
	if (_Count <= _ISORT_MAX)
		_Insertion_sort(_First, _Last);	
	else
		{	
		_Diff _Count2 = (_Count + 1) / 2;
		_BidIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (_Count2 <= _Tempbuf._Maxlen())
			{	
			_Buffered_merge_sort(_First, _Mid, _Count2, _Tempbuf);
			_Buffered_merge_sort(_Mid, _Last, _Count - _Count2, _Tempbuf);
			}
		else
			{	
			_Stable_sort(_First, _Mid, _Count2, _Tempbuf);
			_Stable_sort(_Mid, _Last, _Count - _Count2, _Tempbuf);
			}

		_Buffered_merge(_First, _Mid, _Last,
			_Count2, _Count - _Count2, _Tempbuf);	
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty> inline
	void _Stable_sort(_BidIt _First, _BidIt _Last, _Diff *, _Ty *)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	_Temp_iterator<_Ty> _Tempbuf(_Count);
	_Stable_sort(_First, _Last, _Count, _Tempbuf);
	}

template<class _BidIt> inline
	void stable_sort(_BidIt _First, _BidIt _Last)
	{	
	if (_First != _Last)
		_Stable_sort(_First, _Last, _Dist_type(_First), _Val_type(_First));
	}

		
template<class _BidIt,
	class _OutIt,
	class _Diff,
	class _Pr> inline
	void _Chunked_merge(_BidIt _First, _BidIt _Last, _OutIt _Dest,
		_Diff _Chunk, _Diff _Count, _Pr _Pred)
	{	
	for (_Diff _Chunk2 = _Chunk * 2; _Chunk2 <= _Count; _Count -= _Chunk2)
		{	
		_BidIt _Mid1 = _First;
		std::advance(_Mid1, _Chunk);
		_BidIt _Mid2 = _Mid1;
		std::advance(_Mid2, _Chunk);

		_Dest = std::merge(_First, _Mid1, _Mid1, _Mid2, _Dest, _Pred);
		_First = _Mid2;
		}

	if (_Count <= _Chunk)
		std::copy(_First, _Last, _Dest);	
	else
		{	
		_BidIt _Mid1 = _First;
		std::advance(_Mid1, _Chunk);

		std::merge(_First, _Mid1, _Mid1, _Last, _Dest, _Pred);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Buffered_merge_sort(_BidIt _First, _BidIt _Last, _Diff _Count,
		_Temp_iterator<_Ty>& _Tempbuf, _Pr _Pred)
	{	
	_BidIt _Mid = _First;
	for (_Diff _Nleft = _Count; _ISORT_MAX <= _Nleft; _Nleft -= _ISORT_MAX)
		{	
		_BidIt _Midn = _Mid;
		std::advance(_Midn, (int)_ISORT_MAX);

		_Insertion_sort(_Mid, _Midn, _Pred);
		_Mid = _Midn;
		}
	_Insertion_sort(_Mid, _Last, _Pred);	

	for (_Diff _Chunk = _ISORT_MAX; _Chunk < _Count; _Chunk *= 2)
		{	
		_Chunked_merge(_First, _Last, _Tempbuf._Init(),
			_Chunk, _Count, _Pred);
		_Chunked_merge(_Tempbuf._First(), _Tempbuf._Last(), _First,
			_Chunk *= 2, _Count, _Pred);
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Stable_sort(_BidIt _First, _BidIt _Last, _Diff _Count,
		_Temp_iterator<_Ty>& _Tempbuf, _Pr _Pred)
	{	
	if (_Count <= _ISORT_MAX)
		_Insertion_sort(_First, _Last, _Pred);	
	else
		{	
		_Diff _Count2 = (_Count + 1) / 2;
		_BidIt _Mid = _First;
		std::advance(_Mid, _Count2);

		if (_Count2 <= _Tempbuf._Maxlen())
			{	
			_Buffered_merge_sort(_First, _Mid, _Count2, _Tempbuf, _Pred);
			_Buffered_merge_sort(_Mid, _Last, _Count - _Count2,
				_Tempbuf, _Pred);
			}
		else
			{	
			_Stable_sort(_First, _Mid, _Count2, _Tempbuf, _Pred);
			_Stable_sort(_Mid, _Last, _Count - _Count2, _Tempbuf, _Pred);
			}

		_Buffered_merge(_First, _Mid, _Last,
			_Count2, _Count - _Count2, _Tempbuf, _Pred);	
		}
	}

template<class _BidIt,
	class _Diff,
	class _Ty,
	class _Pr> inline
	void _Stable_sort(_BidIt _First, _BidIt _Last, _Diff *, _Ty *, _Pr _Pred)
	{	
	_Diff _Count = 0;
	_Distance(_First, _Last, _Count);
	_Temp_iterator<_Ty> _Tempbuf(_Count);
	_Stable_sort(_First, _Last, _Count, _Tempbuf, _Pred);
	}

template<class _BidIt,
	class _Pr> inline
	void stable_sort(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	if (_First != _Last)
		_Stable_sort(_First, _Last,
			_Dist_type(_First), _Val_type(_First), _Pred);
	}

		
template<class _RanIt,
	class _Ty> inline
	void _Partial_sort(_RanIt _First, _RanIt _Mid, _RanIt _Last, _Ty *)
	{	
	std::make_heap(_First, _Mid);

	for (_RanIt _Next = _Mid; _Next < _Last; ++_Next)
		if (*_Next < *_First)
			_Pop_heap(_First, _Mid, _Next, _Ty(*_Next),
				_Dist_type(_First));	
	std::sort_heap(_First, _Mid);
	}

template<class _RanIt> inline
	void partial_sort(_RanIt _First, _RanIt _Mid, _RanIt _Last)
	{	
	_Partial_sort(_First, _Mid, _Last, _Val_type(_First));
	}

		
template<class _RanIt,
	class _Ty,
	class _Pr> inline
	void _Partial_sort(_RanIt _First, _RanIt _Mid, _RanIt _Last,
		_Pr _Pred, _Ty *)
	{	
	std::make_heap(_First, _Mid, _Pred);

	for (_RanIt _Next = _Mid; _Next < _Last; ++_Next)
		if (_Pred(*_Next, *_First))
			_Pop_heap(_First, _Mid, _Next, _Ty(*_Next), _Pred,
				_Dist_type(_First));	
	std::sort_heap(_First, _Mid, _Pred);
	}

template<class _RanIt,
	class _Pr> inline
	void partial_sort(_RanIt _First, _RanIt _Mid, _RanIt _Last, _Pr _Pred)
	{	
	_Partial_sort(_First, _Mid, _Last, _Pred, _Val_type(_First));
	}

		
template<class _InIt,
	class _RanIt,
	class _Diff,
	class _Ty> inline
	_RanIt _Partial_sort_copy(_InIt _First1, _InIt _Last1,
		_RanIt _First2, _RanIt _Last2, _Diff *, _Ty *)
	{	
	_RanIt _Mid2 = _First2;
	for (; _First1 != _Last1 && _Mid2 != _Last2; ++_First1, ++_Mid2)
		*_Mid2 = *_First1;	
	std::make_heap(_First2, _Mid2);

	for (; _First1 != _Last1; ++_First1)
		if (*_First1 < *_First2)
			_Adjust_heap(_First2, _Diff(0), _Diff(_Mid2 - _First2),
				_Ty(*_First1));	

	std::sort_heap(_First2, _Mid2);
	return (_Mid2);
	}

template<class _InIt,
	class _RanIt> inline
	_RanIt partial_sort_copy(_InIt _First1, _InIt _Last1,
		_RanIt _First2, _RanIt _Last2)
	{	
	return (_First1 == _Last1 || _First2 == _Last2 ? _First2
		: _Partial_sort_copy(_First1, _Last1, _First2, _Last2,
			_Dist_type(_First2), _Val_type(_First1)));
	}

		
template<class _InIt,
	class _RanIt,
	class _Diff,
	class _Ty, class _Pr> inline
	_RanIt _Partial_sort_copy(_InIt _First1, _InIt _Last1,
		_RanIt _First2, _RanIt _Last2, _Pr _Pred, _Diff *, _Ty *)
	{	
	_RanIt _Mid2 = _First2;
	for (; _First1 != _Last1 && _Mid2 != _Last2; ++_First1, ++_Mid2)
		*_Mid2 = *_First1;	
	std::make_heap(_First2, _Mid2, _Pred);

	for (; _First1 != _Last1; ++_First1)
		if (_Pred(*_First1, *_First2))
			_Adjust_heap(_First2, _Diff(0), _Diff(_Mid2 - _First2),
				_Ty(*_First1), _Pred);	

	std::sort_heap(_First2, _Mid2, _Pred);
	return (_Mid2);
	}

template<class _InIt,
	class _RanIt,
	class _Pr> inline
	_RanIt partial_sort_copy(_InIt _First1, _InIt _Last1,
		_RanIt _First2, _RanIt _Last2, _Pr _Pred)
	{	
	return (_First1 == _Last1 || _First2 == _Last2 ? _First2
		: _Partial_sort_copy(_First1, _Last1, _First2, _Last2, _Pred,
			_Dist_type(_First2), _Val_type(_First1)));
	}

		
template<class _RanIt> inline
	void nth_element(_RanIt _First, _RanIt _Nth, _RanIt _Last)
	{	
	for (; _ISORT_MAX < _Last - _First; )
		{	
		pair<_RanIt, _RanIt> _Mid =
			_Unguarded_partition(_First, _Last);

		if (_Mid.second <= _Nth)
			_First = _Mid.second;
		else if (_Mid.first <= _Nth)
			return;	
		else
			_Last = _Mid.first;
		}

	_Insertion_sort(_First, _Last);	
	}

		
template<class _RanIt,
	class _Pr> inline
	void nth_element(_RanIt _First, _RanIt _Nth, _RanIt _Last, _Pr _Pred)
	{	
	for (; _ISORT_MAX < _Last - _First; )
		{	
		pair<_RanIt, _RanIt> _Mid =
			_Unguarded_partition(_First, _Last, _Pred);

		if (_Mid.second <= _Nth)
			_First = _Mid.second;
		else if (_Mid.first <= _Nth)
			return;	
		else
			_Last = _Mid.first;
		}

	_Insertion_sort(_First, _Last, _Pred);	
	}

		
template<class _InIt1,
	class _InIt2> inline
	bool includes(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (*_First2 < *_First1)
			return (false);
		else if (*_First1 < *_First2)
			++_First1;
		else
			++_First1, ++_First2;
	return (_First2 == _Last2);
	}

		
template<class _InIt1,
	class _InIt2,
	class _Pr> inline
	bool includes(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (_Pred(*_First2, *_First1))
			return (false);
		else if (_Pred(*_First1, *_First2))
			++_First1;
		else
			++_First1, ++_First2;
	return (_First2 == _Last2);
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt> inline
	_OutIt set_union(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (*_First1 < *_First2)
			*_Dest++ = *_First1, ++_First1;
		else if (*_First2 < *_First1)
			*_Dest++ = *_First2, ++_First2;
		else
			*_Dest++ = *_First1, ++_First1, ++_First2;
	_Dest = std::copy(_First1, _Last1, _Dest);
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Pr> inline
	_OutIt set_union(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (_Pred(*_First1, *_First2))
			*_Dest++ = *_First1, ++_First1;
		else if (_Pred(*_First2, *_First1))
			*_Dest++ = *_First2, ++_First2;
		else
			*_Dest++ = *_First1, ++_First1, ++_First2;
	_Dest = std::copy(_First1, _Last1, _Dest);
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt> inline
	_OutIt set_intersection(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (*_First1 < *_First2)
			++_First1;
		else if (*_First2 < *_First1)
			++_First2;
		else
			*_Dest++ = *_First1++, ++_First2;
	return (_Dest);
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Pr> inline
	_OutIt set_intersection(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (_Pred(*_First1, *_First2))
			++_First1;
		else if (_Pred(*_First2, *_First1))
			++_First2;
		else
			*_Dest++ = *_First1++, ++_First2;
	return (_Dest);
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt> inline
	_OutIt set_difference(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2,	_OutIt _Dest)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (*_First1 < *_First2)
			*_Dest++ = *_First1, ++_First1;
		else if (*_First2 < *_First1)
			++_First2;
		else
			++_First1, ++_First2;
	return (std::copy(_First1, _Last1, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Pr> inline
	_OutIt set_difference(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (_Pred(*_First1, *_First2))
			*_Dest++ = *_First1, ++_First1;
		else if (_Pred(*_First2, *_First1))
			++_First2;
		else
			++_First1, ++_First2;
	return (std::copy(_First1, _Last1, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt> inline
	_OutIt set_symmetric_difference(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (*_First1 < *_First2)
			*_Dest++ = *_First1, ++_First1;
		else if (*_First2 < *_First1)
			*_Dest++ = *_First2, ++_First2;
		else
			++_First1, ++_First2;
	_Dest = std::copy(_First1, _Last1, _Dest);
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _InIt1,
	class _InIt2,
	class _OutIt,
	class _Pr> inline
	_OutIt set_symmetric_difference(_InIt1 _First1, _InIt1 _Last1,
		_InIt2 _First2, _InIt2 _Last2, _OutIt _Dest, _Pr _Pred)
	{	
	for (; _First1 != _Last1 && _First2 != _Last2; )
		if (_Pred(*_First1, *_First2))
			*_Dest++ = *_First1, ++_First1;
		else if (_Pred(*_First2, *_First1))
			*_Dest++ = *_First2, ++_First2;
		else
			++_First1, ++_First2;
	_Dest = std::copy(_First1, _Last1, _Dest);
	return (std::copy(_First2, _Last2, _Dest));
	}

		
template<class _FwdIt> inline
	_FwdIt max_element(_FwdIt _First, _FwdIt _Last)
	{	
	_FwdIt _Found = _First;
	if (_First != _Last)
		for (; ++_First != _Last; )
			if (*_Found < *_First)
				_Found = _First;
	return (_Found);
	}

		
template<class _FwdIt,
	class _Pr> inline
	_FwdIt max_element(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	{	
	_FwdIt _Found = _First;
	if (_First != _Last)
		for (; ++_First != _Last; )
			if (_Pred(*_Found, *_First))
				_Found = _First;
	return (_Found);
	}

		
template<class _FwdIt> inline
	_FwdIt min_element(_FwdIt _First, _FwdIt _Last)
	{	
	_FwdIt _Found = _First;
	if (_First != _Last)
		for (; ++_First != _Last; )
			if (*_First < *_Found)
				_Found = _First;
	return (_Found);
	}

		
template<class _FwdIt,
	class _Pr> inline
	_FwdIt min_element(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	{	
	_FwdIt _Found = _First;
	if (_First != _Last)
		for (; ++_First != _Last; )
			if (_Pred(*_First, *_Found))
				_Found = _First;
	return (_Found);
	}

		
template<class _BidIt> inline
	bool next_permutation(_BidIt _First, _BidIt _Last)
	{	
	_BidIt _Next = _Last;
	if (_First == _Last || _First == --_Next)
		return (false);

	for (; ; )
		{	
		_BidIt _Next1 = _Next;
		if (*--_Next < *_Next1)
			{	
			_BidIt _Mid = _Last;
			for (; !(*_Next < *--_Mid); )
				;
			std::iter_swap(_Next, _Mid);
			std::reverse(_Next1, _Last);
			return (true);
			}

		if (_Next == _First)
			{	
			std::reverse(_First, _Last);
			return (false);
			}
		}
	}

		
template<class _BidIt,
	class _Pr> inline
	bool next_permutation(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	_BidIt _Next = _Last;
	if (_First == _Last || _First == --_Next)
		return (false);

	for (; ; )
		{	
		_BidIt _Next1 = _Next;
		if (_Pred(*--_Next, *_Next1))
			{	
			_BidIt _Mid = _Last;
			for (; !_Pred(*_Next, *--_Mid); )
				;
			std::iter_swap(_Next, _Mid);
			std::reverse(_Next1, _Last);
			return (true);
			}

		if (_Next == _First)
			{	
			std::reverse(_First, _Last);
			return (false);
			}
		}
	}

		
template<class _BidIt> inline
	bool prev_permutation(_BidIt _First, _BidIt _Last)
	{	
	_BidIt _Next = _Last;
	if (_First == _Last || _First == --_Next)
		return (false);
	for (; ; )
		{	
		_BidIt _Next1 = _Next;
		if (!(*--_Next < *_Next1))
			{	
			_BidIt _Mid = _Last;
			for (; *_Next < *--_Mid; )
				;
			std::iter_swap(_Next, _Mid);
			std::reverse(_Next1, _Last);
			return (true);
			}

		if (_Next == _First)
			{	
			std::reverse(_First, _Last);
			return (false);
			}
		}
	}

		
template<class _BidIt,
	class _Pr> inline
	bool prev_permutation(_BidIt _First, _BidIt _Last, _Pr _Pred)
	{	
	_BidIt _Next = _Last;
	if (_First == _Last || _First == --_Next)
		return (false);

	for (; ; )
		{	
		_BidIt _Next1 = _Next;
		if (!_Pred(*--_Next, *_Next1))
			{	
			_BidIt _Mid = _Last;
			for (; _Pred(*_Next, *--_Mid); )
				;
			std::iter_swap(_Next, _Mid);
			std::reverse(_Next1, _Last);
			return (true);
			}

		if (_Next == _First)
			{	
			std::reverse(_First, _Last);
			return (false);
			}
		}
	}
}

  #pragma warning(default: 4244)

#pragma warning(pop)
#pragma pack(pop)

#line 2657 "C:\\Program Files\\Microsoft Visual Studio .NET 2003\\VC7\\INCLUDE\\algorithm"






















#line 34 "c:\\boost\\boost/array.hpp"


#line 1 "c:\\boost\\boost/config.hpp"






































































#line 37 "c:\\boost\\boost/array.hpp"


namespace boost {

    template<class T, std::size_t N>
    class array {
      public:
        T elems[N];    

      public:
        
        typedef T              value_type;
        typedef T*             iterator;
        typedef const T*       const_iterator;
        typedef T&             reference;
        typedef const T&       const_reference;
        typedef std::size_t    size_type;
        typedef std::ptrdiff_t difference_type;
    
        
        iterator begin() { return elems; }
        const_iterator begin() const { return elems; }
        iterator end() { return elems+N; }
        const_iterator end() const { return elems+N; }

        

        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;










#line 77 "c:\\boost\\boost/array.hpp"

        reverse_iterator rbegin() { return reverse_iterator(end()); }
        const_reverse_iterator rbegin() const {
            return const_reverse_iterator(end());
        }
        reverse_iterator rend() { return reverse_iterator(begin()); }
        const_reverse_iterator rend() const {
            return const_reverse_iterator(begin());
        }

        
        reference operator[](size_type i) 
        { 
            (void)( (i < N && "out of range") || (_assert("i < N && \"out of range\"", "c:\\boost\\boost/array.hpp", 90), 0) ); 
            return elems[i];
        }
        
        const_reference operator[](size_type i) const 
        {     
            (void)( (i < N && "out of range") || (_assert("i < N && \"out of range\"", "c:\\boost\\boost/array.hpp", 96), 0) ); 
            return elems[i]; 
        }

        
        reference at(size_type i) { rangecheck(i); return elems[i]; }
        const_reference at(size_type i) const { rangecheck(i); return elems[i]; }
    
        
        reference front() 
        { 
            return elems[0]; 
        }
        
        const_reference front() const 
        {
            return elems[0];
        }
        
        reference back() 
        { 
            return elems[N-1]; 
        }
        
        const_reference back() const 
        { 
            return elems[N-1]; 
        }

        
        static size_type size() { return N; }
        static bool empty() { return false; }
        static size_type max_size() { return N; }
        enum { static_size = N };

        
        void swap (array<T,N>& y) {
            std::swap_ranges(begin(),end(),y.begin());
        }

        
        const T* data() const { return elems; }

        
        T* c_array() { return elems; }

        
        template <typename T2>
        array<T,N>& operator= (const array<T2,N>& rhs) {
            std::copy(rhs.begin(),rhs.end(), begin());
            return *this;
        }

        
        void assign (const T& value)
        {
            std::fill_n(begin(),size(),value);
        }

        
        static void rangecheck (size_type i) {
            if (i >= size()) { 
                throw std::range_error("array<>: index out of range");
            }
        }

    };

    
    template<class T, std::size_t N>
    bool operator== (const array<T,N>& x, const array<T,N>& y) {
        return std::equal(x.begin(), x.end(), y.begin());
    }
    template<class T, std::size_t N>
    bool operator< (const array<T,N>& x, const array<T,N>& y) {
        return std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end());
    }
    template<class T, std::size_t N>
    bool operator!= (const array<T,N>& x, const array<T,N>& y) {
        return !(x==y);
    }
    template<class T, std::size_t N>
    bool operator> (const array<T,N>& x, const array<T,N>& y) {
        return y<x;
    }
    template<class T, std::size_t N>
    bool operator<= (const array<T,N>& x, const array<T,N>& y) {
        return !(y<x);
    }
    template<class T, std::size_t N>
    bool operator>= (const array<T,N>& x, const array<T,N>& y) {
        return !(x<y);
    }

    
    template<class T, std::size_t N>
    inline void swap (array<T,N>& x, array<T,N>& y) {
        x.swap(y);
    }

} 

#line 199 "c:\\boost\\boost/array.hpp"
#line 8 "..\\..\\..\\..\\boost/sequence/fixed_size/is_fixed_size.hpp"
#line 1 "c:\\boost\\boost/mpl/bool.hpp"
















#line 1 "c:\\boost\\boost/mpl/bool_fwd.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/msvc.hpp"


















#line 1 "c:\\boost\\boost/config.hpp"






































































#line 20 "c:\\boost\\boost/mpl/aux_/config/msvc.hpp"

#line 22 "c:\\boost\\boost/mpl/aux_/config/msvc.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/intel.hpp"


















#line 1 "c:\\boost\\boost/config.hpp"






































































#line 20 "c:\\boost\\boost/mpl/aux_/config/intel.hpp"

#line 22 "c:\\boost\\boost/mpl/aux_/config/intel.hpp"
#line 19 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"


















#line 20 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"

#line 22 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"

#line 24 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"
#line 20 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"
















#line 1 "c:\\boost\\boost/detail/workaround.hpp"
































































#line 66 "c:\\boost\\boost/detail/workaround.hpp"





#line 72 "c:\\boost\\boost/detail/workaround.hpp"

#line 74 "c:\\boost\\boost/detail/workaround.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"

#line 20 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"
#line 21 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"











#line 39 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"

#line 41 "c:\\boost\\boost/mpl/aux_/config/adl.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"























#line 19 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 20 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"



















#line 40 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"






#line 47 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"

#line 49 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
#line 18 "c:\\boost\\boost/mpl/bool_fwd.hpp"

namespace boost { namespace mpl {

template< bool C_ > struct bool_;


typedef bool_<true> true_;
typedef bool_<false> false_;

}}





#line 34 "c:\\boost\\boost/mpl/bool_fwd.hpp"
#line 18 "c:\\boost\\boost/mpl/bool.hpp"
#line 1 "c:\\boost\\boost/mpl/integral_c_tag.hpp"

















#line 1 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
















































#line 19 "c:\\boost\\boost/mpl/integral_c_tag.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"


















#line 1 "c:\\boost\\boost/config.hpp"






































































#line 20 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"



#line 24 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"

#line 26 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"
#line 20 "c:\\boost\\boost/mpl/integral_c_tag.hpp"

namespace boost { namespace mpl {
struct integral_c_tag { static const int value = 0; };
}}


#line 27 "c:\\boost\\boost/mpl/integral_c_tag.hpp"
#line 19 "c:\\boost\\boost/mpl/bool.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"

























#line 20 "c:\\boost\\boost/mpl/bool.hpp"

namespace boost { namespace mpl {

template< bool C_ > struct bool_
{
    static const bool value = C_;
    typedef integral_c_tag tag;
    typedef bool_ type;
    typedef bool value_type;
    operator bool() const { return this->value; }
};


template< bool C_ >
bool const bool_<C_>::value;
#line 36 "c:\\boost\\boost/mpl/bool.hpp"

}}

#line 40 "c:\\boost\\boost/mpl/bool.hpp"
#line 9 "..\\..\\..\\..\\boost/sequence/fixed_size/is_fixed_size.hpp"


namespace boost { namespace sequence { namespace fixed_size {

template <class T>
struct is_fixed_size
  : mpl::false_ {};

template <class T, std::size_t N>
struct is_fixed_size<T[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T(&)[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T const[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T const(&)[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<boost::array<T,N> >
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<boost::array<T,N> const>
  : mpl::true_ {};

}}} 

#line 44 "..\\..\\..\\..\\boost/sequence/fixed_size/is_fixed_size.hpp"
#line 9 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"
#line 1 "..\\..\\..\\..\\boost/sequence/algorithm/fixed_size/category.hpp"






namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

struct category {};

}}}} 

#line 14 "..\\..\\..\\..\\boost/sequence/algorithm/fixed_size/category.hpp"
#line 10 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"
#line 1 "c:\\boost\\boost/utility/enable_if.hpp"
















#line 1 "c:\\boost\\boost/config.hpp"






































































#line 18 "c:\\boost\\boost/utility/enable_if.hpp"






namespace boost
{
 
  template <bool B, class T = void>
  struct enable_if_c {
    typedef T type;
  };

  template <class T>
  struct enable_if_c<false, T> {};

  template <class Cond, class T = void> 
  struct enable_if : public enable_if_c<Cond::value, T> {};

  template <bool B, class T>
  struct lazy_enable_if_c {
    typedef typename T::type type;
  };

  template <class T>
  struct lazy_enable_if_c<false, T> {};

  template <class Cond, class T> 
  struct lazy_enable_if : public lazy_enable_if_c<Cond::value, T> {};


  template <bool B, class T = void>
  struct disable_if_c {
    typedef T type;
  };

  template <class T>
  struct disable_if_c<true, T> {};

  template <class Cond, class T = void> 
  struct disable_if : public disable_if_c<Cond::value, T> {};

  template <bool B, class T>
  struct lazy_disable_if_c {
    typedef typename T::type type;
  };

  template <class T>
  struct lazy_disable_if_c<true, T> {};

  template <class Cond, class T> 
  struct lazy_disable_if : public lazy_disable_if_c<Cond::value, T> {};

} 












































#line 118 "c:\\boost\\boost/utility/enable_if.hpp"

#line 120 "c:\\boost\\boost/utility/enable_if.hpp"
#line 11 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"

namespace boost { namespace sequence { 

template <class Sequence>
struct category<
    Sequence
  , typename enable_if<fixed_size::is_fixed_size<Sequence> >::type
>
{
    typedef algorithm::fixed_size::category type;
};

}} 

#line 26 "..\\..\\..\\..\\boost/sequence/fixed_size/category.hpp"
#line 8 "..\\..\\..\\..\\boost/sequence/category.hpp"

#line 10 "..\\..\\..\\..\\boost/sequence/category.hpp"
#line 8 "..\\..\\..\\..\\boost/sequence/algorithm/dispatch.hpp"
#line 1 "..\\..\\..\\..\\boost/typeof/typeof.hpp"







#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























#line 30 "c:\\boost\\boost/preprocessor/config/config.hpp"

#line 32 "c:\\boost\\boost/preprocessor/config/config.hpp"

#line 34 "c:\\boost\\boost/preprocessor/config/config.hpp"

#line 36 "c:\\boost\\boost/preprocessor/config/config.hpp"

#line 38 "c:\\boost\\boost/preprocessor/config/config.hpp"



#line 42 "c:\\boost\\boost/preprocessor/config/config.hpp"
#line 43 "c:\\boost\\boost/preprocessor/config/config.hpp"





#line 49 "c:\\boost\\boost/preprocessor/config/config.hpp"








#line 58 "c:\\boost\\boost/preprocessor/config/config.hpp"
#line 59 "c:\\boost\\boost/preprocessor/config/config.hpp"

#line 61 "c:\\boost\\boost/preprocessor/config/config.hpp"
#line 18 "c:\\boost\\boost/preprocessor/cat.hpp"








#line 27 "c:\\boost\\boost/preprocessor/cat.hpp"



#line 31 "c:\\boost\\boost/preprocessor/cat.hpp"


#line 34 "c:\\boost\\boost/preprocessor/cat.hpp"

#line 36 "c:\\boost\\boost/preprocessor/cat.hpp"
#line 9 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/cat.hpp"














#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 16 "c:\\boost\\boost/preprocessor/seq/cat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"














#line 1 "c:\\boost\\boost/preprocessor/arithmetic/dec.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/arithmetic/dec.hpp"








#line 27 "c:\\boost\\boost/preprocessor/arithmetic/dec.hpp"





































































































































































































































































#line 289 "c:\\boost\\boost/preprocessor/arithmetic/dec.hpp"
#line 16 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 17 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/control/if.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/control/if.hpp"
#line 1 "c:\\boost\\boost/preprocessor/control/iif.hpp"














#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 16 "c:\\boost\\boost/preprocessor/control/iif.hpp"






#line 23 "c:\\boost\\boost/preprocessor/control/iif.hpp"



#line 27 "c:\\boost\\boost/preprocessor/control/iif.hpp"


#line 30 "c:\\boost\\boost/preprocessor/control/iif.hpp"




#line 35 "c:\\boost\\boost/preprocessor/control/iif.hpp"
#line 19 "c:\\boost\\boost/preprocessor/control/if.hpp"
#line 1 "c:\\boost\\boost/preprocessor/logical/bool.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/logical/bool.hpp"








#line 27 "c:\\boost\\boost/preprocessor/logical/bool.hpp"





































































































































































































































































#line 289 "c:\\boost\\boost/preprocessor/logical/bool.hpp"
#line 20 "c:\\boost\\boost/preprocessor/control/if.hpp"








#line 29 "c:\\boost\\boost/preprocessor/control/if.hpp"

#line 31 "c:\\boost\\boost/preprocessor/control/if.hpp"
#line 18 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/debug/error.hpp"














#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 16 "c:\\boost\\boost/preprocessor/debug/error.hpp"
#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 17 "c:\\boost\\boost/preprocessor/debug/error.hpp"





#line 23 "c:\\boost\\boost/preprocessor/debug/error.hpp"










#line 34 "c:\\boost\\boost/preprocessor/debug/error.hpp"
#line 19 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"











#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 13 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"



#line 17 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"




#line 1 "c:\\boost\\boost/preprocessor/control/iif.hpp"


































#line 22 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"














































































































































































































































































#line 293 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"
#line 294 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"
#line 20 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/seq.hpp"














#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 16 "c:\\boost\\boost/preprocessor/seq/seq.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/elem.hpp"














#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 16 "c:\\boost\\boost/preprocessor/seq/elem.hpp"
#line 1 "c:\\boost\\boost/preprocessor/facilities/empty.hpp"




















#line 22 "c:\\boost\\boost/preprocessor/facilities/empty.hpp"
#line 17 "c:\\boost\\boost/preprocessor/seq/elem.hpp"







#line 25 "c:\\boost\\boost/preprocessor/seq/elem.hpp"















#line 41 "c:\\boost\\boost/preprocessor/seq/elem.hpp"


































































































































































































































































#line 300 "c:\\boost\\boost/preprocessor/seq/elem.hpp"
#line 17 "c:\\boost\\boost/preprocessor/seq/seq.hpp"











#line 29 "c:\\boost\\boost/preprocessor/seq/seq.hpp"







#line 37 "c:\\boost\\boost/preprocessor/seq/seq.hpp"







#line 45 "c:\\boost\\boost/preprocessor/seq/seq.hpp"
#line 21 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/size.hpp"














#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 16 "c:\\boost\\boost/preprocessor/seq/size.hpp"
#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 17 "c:\\boost\\boost/preprocessor/seq/size.hpp"
#line 1 "c:\\boost\\boost/preprocessor/tuple/eat.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/tuple/eat.hpp"








#line 27 "c:\\boost\\boost/preprocessor/tuple/eat.hpp"






























#line 58 "c:\\boost\\boost/preprocessor/tuple/eat.hpp"
#line 18 "c:\\boost\\boost/preprocessor/seq/size.hpp"





#line 24 "c:\\boost\\boost/preprocessor/seq/size.hpp"




#line 29 "c:\\boost\\boost/preprocessor/seq/size.hpp"





































































































































































































































































































































































































































































































































#line 547 "c:\\boost\\boost/preprocessor/seq/size.hpp"
#line 22 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"





#line 28 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1069 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"

#line 1071 "c:\\boost\\boost/preprocessor/seq/fold_left.hpp"
#line 17 "c:\\boost\\boost/preprocessor/seq/cat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/seq/seq.hpp"












































#line 18 "c:\\boost\\boost/preprocessor/seq/cat.hpp"








#line 27 "c:\\boost\\boost/preprocessor/seq/cat.hpp"











#line 39 "c:\\boost\\boost/preprocessor/seq/cat.hpp"

#line 41 "c:\\boost\\boost/preprocessor/seq/cat.hpp"
#line 10 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 1 "c:\\boost\\boost/preprocessor/expand.hpp"














#line 1 "c:\\boost\\boost/preprocessor/facilities/expand.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/facilities/expand.hpp"






#line 25 "c:\\boost\\boost/preprocessor/facilities/expand.hpp"



#line 29 "c:\\boost\\boost/preprocessor/facilities/expand.hpp"
#line 16 "c:\\boost\\boost/preprocessor/expand.hpp"

#line 18 "c:\\boost\\boost/preprocessor/expand.hpp"
#line 11 "..\\..\\..\\..\\boost/typeof/typeof.hpp"



#line 1 "..\\..\\..\\..\\boost/typeof/config.hpp"







#line 1 "c:\\boost\\boost/config.hpp"






































































#line 9 "..\\..\\..\\..\\boost/typeof/config.hpp"






#line 19 "..\\..\\..\\..\\boost/typeof/config.hpp"


#line 22 "..\\..\\..\\..\\boost/typeof/config.hpp"









#line 32 "..\\..\\..\\..\\boost/typeof/config.hpp"

#line 34 "..\\..\\..\\..\\boost/typeof/config.hpp"






#line 41 "..\\..\\..\\..\\boost/typeof/config.hpp"
#line 15 "..\\..\\..\\..\\boost/typeof/typeof.hpp"





#line 1 "..\\..\\..\\..\\boost/typeof/message.hpp"





#pragma message("using msvc 'native' imlementation")
#line 8 "..\\..\\..\\..\\boost/typeof/message.hpp"

#line 21 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 1 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"
                                                                                                                                                                                                                                                               









#line 1 "c:\\boost\\boost/config.hpp"






































































#line 12 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"
#line 1 "c:\\boost\\boost/detail/workaround.hpp"









































































#line 13 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"

namespace boost
{
    namespace type_of
    {

        

        template<int N> struct the_counter;
        
        template<typename T,int N = 5>
        struct encode_counter
        {
            __if_exists(the_counter<N + 256>)
            {
                static const unsigned count=(encode_counter<T,N + 257>::count);
            }
            __if_not_exists(the_counter<N + 256>)
            {
                __if_exists(the_counter<N + 64>)
                {
                    static const unsigned count=(encode_counter<T,N + 65>::count);
                }
                __if_not_exists(the_counter<N + 64>)
                {
                    __if_exists(the_counter<N + 16>)
                    { 
                        static const unsigned count=(encode_counter<T,N + 17>::count);
                    }
                    __if_not_exists(the_counter<N + 16>)
                    {
                        __if_exists(the_counter<N + 4>)
                        {
                            static const unsigned count=(encode_counter<T,N + 5>::count);
                        }
                        __if_not_exists(the_counter<N + 4>)
                        {
                            __if_exists(the_counter<N>)
                            {
                                static const unsigned count=(encode_counter<T,N + 1>::count);
                            }
                            __if_not_exists(the_counter<N>)
                            {
                                static const unsigned count=N;
                                typedef the_counter<N> type;
                            }
                        }
                    }
                }
            }
        };












#line 77 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"

        




















#line 100 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"
        template<int ID>
        struct msvc_typeof_base
        {

            struct id2type;
        };

        template<typename T, int ID>
        struct msvc_typeof : msvc_typeof_base<ID>
        {
            struct msvc_typeof_base<ID>::id2type 
            {
                typedef T type;
            };
        };
#line 116 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"
        template<int ID>
        struct msvc_typeid_wrapper {
            typedef typename msvc_typeof_base<ID>::id2type id2type;
            typedef typename id2type::type type;
        };
        
        template<>
        struct msvc_typeid_wrapper<1> {
            typedef msvc_typeid_wrapper<1> type;
        };
        
        template<>
        struct msvc_typeid_wrapper<4> {
            typedef msvc_typeid_wrapper<4> type;
        };

        
        template<typename T>
        struct encode_type
        {
            
            static const unsigned value=(encode_counter<T>::count);
            
            typedef typename msvc_typeof<T,value>::id2type type;
            
            static const unsigned next=value+1;
            
            ;     
        };

        template<typename T>
        char (*encode_start(T const&))[encode_type<T>::value];
    }
}






#line 157 "..\\..\\..\\..\\boost/typeof/msvc/typeof_impl.hpp"
#line 22 "..\\..\\..\\..\\boost/typeof/typeof.hpp"


















#line 41 "..\\..\\..\\..\\boost/typeof/typeof.hpp"









#line 51 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 1 "..\\..\\..\\..\\boost/typeof/compliant/lvalue_typeof.hpp"







#line 1 "c:\\boost\\boost/type_traits/is_const.hpp"























#line 1 "c:\\boost\\boost/config.hpp"






































































#line 25 "c:\\boost\\boost/type_traits/is_const.hpp"


#line 1 "c:\\boost\\boost/type_traits/detail/cv_traits_impl.hpp"













#line 1 "c:\\boost\\boost/config.hpp"






































































#line 15 "c:\\boost\\boost/type_traits/detail/cv_traits_impl.hpp"



namespace boost {
namespace detail {



template <typename T> struct cv_traits_imp {};

template <typename T>
struct cv_traits_imp<T*>
{
    static const bool is_const = false;
    static const bool is_volatile = false;
    typedef T unqualified_type;
};

template <typename T>
struct cv_traits_imp<const T*>
{
    static const bool is_const = true;
    static const bool is_volatile = false;
    typedef T unqualified_type;
};

template <typename T>
struct cv_traits_imp<volatile T*>
{
    static const bool is_const = false;
    static const bool is_volatile = true;
    typedef T unqualified_type;
};

template <typename T>
struct cv_traits_imp<const volatile T*>
{
    static const bool is_const = true;
    static const bool is_volatile = true;
    typedef T unqualified_type;
};

} 
} 

#line 61 "c:\\boost\\boost/type_traits/detail/cv_traits_impl.hpp"

#line 63 "c:\\boost\\boost/type_traits/detail/cv_traits_impl.hpp"
#line 28 "c:\\boost\\boost/type_traits/is_const.hpp"








#line 37 "c:\\boost\\boost/type_traits/is_const.hpp"


#line 1 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"













#line 1 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"









#line 1 "c:\\boost\\boost/mpl/int.hpp"
















#line 1 "c:\\boost\\boost/mpl/int_fwd.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
















































#line 18 "c:\\boost\\boost/mpl/int_fwd.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/nttp.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/msvc.hpp"





















#line 18 "c:\\boost\\boost/mpl/aux_/config/nttp.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 19 "c:\\boost\\boost/mpl/aux_/config/nttp.hpp"


















#line 40 "c:\\boost\\boost/mpl/aux_/config/nttp.hpp"

#line 42 "c:\\boost\\boost/mpl/aux_/config/nttp.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"











#line 30 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"



#line 34 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"

#line 36 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"
#line 19 "c:\\boost\\boost/mpl/int_fwd.hpp"

namespace boost { namespace mpl {

template< int N > struct int_;

}}


#line 28 "c:\\boost\\boost/mpl/int_fwd.hpp"
#line 18 "c:\\boost\\boost/mpl/int.hpp"


#line 1 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"















#line 1 "c:\\boost\\boost/mpl/integral_c_tag.hpp"


























#line 17 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 18 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"



#line 24 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"

#line 26 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"

#line 28 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"



































#line 19 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"

























#line 20 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 21 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 23 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"



#line 27 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"



#line 31 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"




#line 36 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

#line 38 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 39 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

namespace boost { namespace mpl {

template< int N >
struct int_
{
    static const int value = N;




#line 51 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef int_ type;
#line 53 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef int value_type;
    typedef integral_c_tag tag;











#line 67 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"


#line 72 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef boost::mpl::int_< static_cast<int>((value + 1)) > next;
    typedef boost::mpl::int_< static_cast<int>((value - 1)) > prior;
#line 75 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

    
    
    
    
    operator int() const { return static_cast<int>(this->value); } 
};


template< int N >
int const boost::mpl::int_< N >::value;
#line 87 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

}}





#line 21 "c:\\boost\\boost/mpl/int.hpp"

#line 23 "c:\\boost\\boost/mpl/int.hpp"
#line 11 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/template_arity_fwd.hpp"
















namespace boost { namespace mpl { namespace aux {

template< typename F > struct template_arity;

}}}

#line 24 "c:\\boost\\boost/mpl/aux_/template_arity_fwd.hpp"
#line 12 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 18 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"





#line 28 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"



#line 32 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"



#line 37 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"


#line 40 "c:\\boost\\boost/mpl/aux_/config/preprocessor.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"

























#line 44 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"

#line 1 "c:\\boost\\boost/preprocessor/comma_if.hpp"














#line 1 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"
#line 1 "c:\\boost\\boost/preprocessor/control/if.hpp"






























#line 19 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"
#line 1 "c:\\boost\\boost/preprocessor/facilities/empty.hpp"





















#line 20 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"
#line 1 "c:\\boost\\boost/preprocessor/punctuation/comma.hpp"




















#line 22 "c:\\boost\\boost/preprocessor/punctuation/comma.hpp"
#line 21 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"








#line 30 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"

#line 32 "c:\\boost\\boost/preprocessor/punctuation/comma_if.hpp"
#line 16 "c:\\boost\\boost/preprocessor/comma_if.hpp"

#line 18 "c:\\boost\\boost/preprocessor/comma_if.hpp"
#line 46 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"
#line 1 "c:\\boost\\boost/preprocessor/repeat.hpp"














#line 1 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 18 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 19 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/debug/error.hpp"

































#line 20 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"











#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 13 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"



#line 17 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"




















































































































































































































































































#line 294 "c:\\boost\\boost/preprocessor/detail/auto_rec.hpp"
#line 21 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
#line 1 "c:\\boost\\boost/preprocessor/tuple/eat.hpp"

























































#line 22 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"





#line 28 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"





























































































































































































































































































































































































































































































































































































































































































































































































































#line 826 "c:\\boost\\boost/preprocessor/repetition/repeat.hpp"
#line 16 "c:\\boost\\boost/preprocessor/repeat.hpp"

#line 18 "c:\\boost\\boost/preprocessor/repeat.hpp"
#line 47 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"
#line 1 "c:\\boost\\boost/preprocessor/inc.hpp"














#line 1 "c:\\boost\\boost/preprocessor/arithmetic/inc.hpp"
















#line 1 "c:\\boost\\boost/preprocessor/config/config.hpp"




























































#line 18 "c:\\boost\\boost/preprocessor/arithmetic/inc.hpp"








#line 27 "c:\\boost\\boost/preprocessor/arithmetic/inc.hpp"





































































































































































































































































#line 289 "c:\\boost\\boost/preprocessor/arithmetic/inc.hpp"
#line 16 "c:\\boost\\boost/preprocessor/inc.hpp"

#line 18 "c:\\boost\\boost/preprocessor/inc.hpp"
#line 48 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"
#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 49 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"














#line 64 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"

#line 66 "c:\\boost\\boost/mpl/aux_/preprocessor/params.hpp"
#line 13 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/msvc.hpp"





















#line 18 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/gcc.hpp"























#line 19 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 20 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"





#line 27 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"






#line 38 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"

#line 40 "c:\\boost\\boost/mpl/aux_/config/ttp.hpp"
#line 18 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 18 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"
#line 1 "c:\\boost\\boost/config.hpp"






































































#line 19 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"





#line 27 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"



#line 31 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"
#line 19 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"








#line 31 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"

#line 33 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"
#line 14 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/overload_resolution.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 18 "c:\\boost\\boost/mpl/aux_/config/overload_resolution.hpp"





#line 28 "c:\\boost\\boost/mpl/aux_/config/overload_resolution.hpp"

#line 30 "c:\\boost\\boost/mpl/aux_/config/overload_resolution.hpp"
#line 15 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"













#line 30 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"

#line 32 "c:\\boost\\boost/type_traits/detail/template_arity_spec.hpp"
#line 15 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"
#line 1 "c:\\boost\\boost/type_traits/integral_constant.hpp"








#line 1 "c:\\boost\\boost/config.hpp"






































































#line 10 "c:\\boost\\boost/type_traits/integral_constant.hpp"
#line 1 "c:\\boost\\boost/mpl/bool.hpp"







































#line 11 "c:\\boost\\boost/type_traits/integral_constant.hpp"
#line 1 "c:\\boost\\boost/mpl/integral_c.hpp"
















#line 1 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 18 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/adl_barrier.hpp"
















































#line 19 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"

namespace boost { namespace mpl {




#line 26 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"
template< typename T, T N > struct integral_c;
#line 28 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"

}}


#line 33 "c:\\boost\\boost/mpl/integral_c_fwd.hpp"
#line 18 "c:\\boost\\boost/mpl/integral_c.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/ctps.hpp"






























#line 19 "c:\\boost\\boost/mpl/integral_c.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"

























#line 20 "c:\\boost\\boost/mpl/integral_c.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 21 "c:\\boost\\boost/mpl/integral_c.hpp"




#line 26 "c:\\boost\\boost/mpl/integral_c.hpp"

#line 28 "c:\\boost\\boost/mpl/integral_c.hpp"




#line 1 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"















#line 1 "c:\\boost\\boost/mpl/integral_c_tag.hpp"


























#line 17 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/static_cast.hpp"



























#line 18 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/nttp_decl.hpp"



































#line 19 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/static_constant.hpp"

























#line 20 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/config/workaround.hpp"



















#line 21 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 23 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"



#line 27 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"



#line 31 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"







#line 39 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

namespace boost { namespace mpl {

template< typename T, T N >
struct integral_c
{
    static const T value = N;




#line 51 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef integral_c type;
#line 53 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef T value_type;
    typedef integral_c_tag tag;











#line 67 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"


#line 72 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"
    typedef integral_c< T, static_cast<T>((value + 1)) > next;
    typedef integral_c< T, static_cast<T>((value - 1)) > prior;
#line 75 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

    
    
    
    
    operator T() const { return static_cast<T>(this->value); } 
};


template< typename T, T N >
T const integral_c< T, N >::value;
#line 87 "c:\\boost\\boost/mpl/aux_/integral_wrapper.hpp"

}}





#line 33 "c:\\boost\\boost/mpl/integral_c.hpp"



namespace boost { namespace mpl {

template< bool C >
struct integral_c<bool, C>
{
    static const bool value = C;
    typedef integral_c type;
    operator bool() const { return this->value; }
};
}}
#line 48 "c:\\boost\\boost/mpl/integral_c.hpp"

#line 50 "c:\\boost\\boost/mpl/integral_c.hpp"
#line 12 "c:\\boost\\boost/type_traits/integral_constant.hpp"

namespace boost{



#line 18 "c:\\boost\\boost/type_traits/integral_constant.hpp"
template <class T, T val>
#line 20 "c:\\boost\\boost/type_traits/integral_constant.hpp"
struct integral_constant : public mpl::integral_c<T, val>
{
   
   
   typedef integral_constant<T,val> type;
































#line 58 "c:\\boost\\boost/type_traits/integral_constant.hpp"
};

template<> struct integral_constant<bool,true> : public mpl::true_ 
{



#line 66 "c:\\boost\\boost/type_traits/integral_constant.hpp"
   typedef integral_constant<bool,true> type;
};
template<> struct integral_constant<bool,false> : public mpl::false_ 
{



#line 74 "c:\\boost\\boost/type_traits/integral_constant.hpp"
   typedef integral_constant<bool,false> type;
};

typedef integral_constant<bool,true> true_type;
typedef integral_constant<bool,false> false_type;

}

#line 83 "c:\\boost\\boost/type_traits/integral_constant.hpp"
#line 16 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"
#line 1 "c:\\boost\\boost/mpl/bool.hpp"







































#line 17 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"
#line 1 "c:\\boost\\boost/mpl/aux_/lambda_support.hpp"
















#line 1 "c:\\boost\\boost/mpl/aux_/config/lambda.hpp"
































#line 18 "c:\\boost\\boost/mpl/aux_/lambda_support.hpp"





















































































































































#line 168 "c:\\boost\\boost/mpl/aux_/lambda_support.hpp"

#line 170 "c:\\boost\\boost/mpl/aux_/lambda_support.hpp"
#line 18 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"
#line 1 "c:\\boost\\boost/config.hpp"






































































#line 19 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"








#line 28 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"






#line 35 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"



#line 39 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"



#line 43 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"












































































































#line 152 "c:\\boost\\boost/type_traits/detail/bool_trait_def.hpp"
#line 40 "c:\\boost\\boost/type_traits/is_const.hpp"

namespace boost {




template< typename T > struct is_const : ::boost::integral_constant<bool,::boost::detail::cv_traits_imp<T*>::is_const> {   }; 
template< typename T > struct is_const< T& > : ::boost::integral_constant<bool,false> {  };









#line 58 "c:\\boost\\boost/type_traits/is_const.hpp"






#line 65 "c:\\boost\\boost/type_traits/is_const.hpp"






























































#line 128 "c:\\boost\\boost/type_traits/is_const.hpp"

} 

#line 1 "c:\\boost\\boost/type_traits/detail/bool_trait_undef.hpp"



























#line 132 "c:\\boost\\boost/type_traits/is_const.hpp"

#line 134 "c:\\boost\\boost/type_traits/is_const.hpp"

#line 9 "..\\..\\..\\..\\boost/typeof/compliant/lvalue_typeof.hpp"

namespace boost
{
    namespace type_of
    {
        enum
        {
            RVALUE = 1,
            LVALUE,
            CONST_LVALUE
        };

        char(&classify_expression(...))[
            RVALUE
        ];

        template<class T>
        char(&classify_expression(T&))[
            is_const<T>::value ? CONST_LVALUE : LVALUE
        ];

        template<class T, int n> struct decorate_type
        {
            typedef T type;
        };
        template<class T> struct decorate_type<T, LVALUE>
        {
            typedef T& type;
        };
        template<class T> struct decorate_type<T, CONST_LVALUE>
        {
            typedef const T& type;
        };
    }
}










#line 55 "..\\..\\..\\..\\boost/typeof/compliant/lvalue_typeof.hpp"
#line 52 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 53 "..\\..\\..\\..\\boost/typeof/typeof.hpp"






#line 60 "..\\..\\..\\..\\boost/typeof/typeof.hpp"


#line 63 "..\\..\\..\\..\\boost/typeof/typeof.hpp"


#line 66 "..\\..\\..\\..\\boost/typeof/typeof.hpp"








#line 1 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"







#line 1 "..\\..\\..\\..\\boost/typeof/typeof.hpp"



























































































#line 9 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"

#line 1 "..\\..\\..\\..\\boost/typeof/increment_registration_group.hpp"










#line 1 "c:\\boost\\boost/preprocessor/slot/slot.hpp"














#line 1 "c:\\boost\\boost/preprocessor/cat.hpp"



































#line 16 "c:\\boost\\boost/preprocessor/slot/slot.hpp"
#line 1 "c:\\boost\\boost/preprocessor/slot/detail/def.hpp"
















































#line 50 "c:\\boost\\boost/preprocessor/slot/detail/def.hpp"
#line 17 "c:\\boost\\boost/preprocessor/slot/slot.hpp"















#line 33 "c:\\boost\\boost/preprocessor/slot/slot.hpp"
#line 12 "..\\..\\..\\..\\boost/typeof/increment_registration_group.hpp"









#line 22 "..\\..\\..\\..\\boost/typeof/increment_registration_group.hpp"
#line 11 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"























#line 35 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"












#line 50 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"







#line 58 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"





#line 64 "..\\..\\..\\..\\boost/typeof/register_fundamental.hpp"
#line 75 "..\\..\\..\\..\\boost/typeof/typeof.hpp"








#line 84 "..\\..\\..\\..\\boost/typeof/typeof.hpp"




#line 89 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 90 "..\\..\\..\\..\\boost/typeof/typeof.hpp"

#line 92 "..\\..\\..\\..\\boost/typeof/typeof.hpp"
#line 9 "..\\..\\..\\..\\boost/sequence/algorithm/dispatch.hpp"

namespace boost { namespace sequence { namespace algorithm { 













template <class Signature> struct dispatch;

template <class T>


template <class AlgorithmID, class Cat1, class Cat2>
struct get_impl
  : boost::type_of::msvc_typeid_wrapper<sizeof(*boost::type_of::encode_start(lookup_implementation(AlgorithmID(), cat1(), cat2())))>::type
#line 33 "..\\..\\..\\..\\boost/sequence/algorithm/dispatch.hpp"
{};

}}} 
