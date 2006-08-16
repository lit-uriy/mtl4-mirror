/*****************************************************************************
  file: memory.h
  -------------- 
  Defines the class Memory to take care of baseBlocks and quadNodes memory 
    allocation needs. 
  The idea is to dynamically allocate sets of baseBlocks (or nodes) 
    under an as needed basis. 
  Here, "stock" reffers to allocated set of a given data type.
  use an aSpace* "storage" to collect all allocated stocks.

  Revised on: 07/25/06
  
  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef MEMORY_H
#define MEMORY_H

#include <cstdlib>
#include <malloc.h>
#include "defs.h"

#define MEM_STEP 0x100000	  // allocate 1 MB of base blocks at the same time.

// aSpace --> data type of the storage of stocks.
typedef struct sp {
	void* space;     //ptr to allocated space
	sp* prev;        //previous ptr
} aSpace;

extern int alloc_step;  // numb of baseBlocks to allocate at a time
                          // = numb of other data types to allocate at a time
                          //--> declared as external since the function 
                          //    that computes it must be inlined 
                          //    for optimization purposes.

/*class Memory
  type template T : data type used by the calling instance to allocate memory.
                    data type must have same size as void* due to the way 
                    fromMem() and toMem(T**) behave. (see defs below). 
  
  Notice: 
    class used to group memory-related global data together. 
   
    (?)member variables are declared public due to the need to impose the 
      inlining of functions that use them. (ex: in stock.h).
      
    setting up a stock of T's:
      --> call constructor
      --> set step
      --> set alloc-step
      --> set max_alloc_reqd
      --> call init();   
*/
template <class T>
class Memory
{
private:
  //Set the stock capacity and update the maximum number of allocation that
  //  we will ever need for the given data type.  
	void set_stock_capacity() {
	
	  if(max_alloc_reqd <= 0) 
	     printf("*WARNING*:  "
	           "initMem():set_stock_capacity(): max-alloc_reqd is null!!!\n");
	  if(stock_step <= 0) 
	     printf("*WARNING*:  "
	           "initMem():set_stock_capacity(): stock_step is null!!!\n");
	           
		stock_capacity = min((indexType)alloc_step, max_alloc_reqd);
		max_alloc_reqd -= stock_capacity;
		stock_capacity *= stock_step; 
	}
public:
	T* stock_head;       //pointer to head of stock of T's
	T* stock_tail;       //pointer to end of stock of T's
	T* avail;            //pointer to available part of stock of T's
	indexType max_alloc_reqd;  //max numb of T's ever needed
	indexType stock_capacity;  //numb of T's to allocate currently.
	int stock_step;      //numb of T's to get at a time in fromMem().
	aSpace* spaces;      //pointer to the storage of stocks of T's

  //default constructor: initializes max_alloc_reqd to 0, and spaces to NIL
	Memory() : max_alloc_reqd( 0 ), spaces( NIL ) {}
	
	//Constructor Memory(int step): same as default constructor. 
    //Also initializes stock_step to step.
	Memory(int step): max_alloc_reqd( 0 ), stock_step( step ), spaces( NIL ) {}
	
	//increment max_alloc_reqd
	void max_alloc_requirement(int alloc_inc) {
	  indexType t = max_alloc_reqd + alloc_inc;
	  if(t>max_alloc_reqd)
	    t = std::numeric_limits<indexType>::max();
		max_alloc_reqd = t;
	}
	
	//set stock_step
	void set_stock_step(int step) { stock_step = step; }
	
	//set the allocated stock ready for memory-related demands
	void initMem() {
    //update the stock capacity and max_alloc_reqd. 
		set_stock_capacity();
		
		//new stock ==> initialize stock_head, stock_tail, and avail
		stock_head = (T*)valloc(stock_capacity*sizeof(T));
		stock_tail = stock_head + stock_capacity;
		avail = stock_head; 
		
		//append this stock to stored stocks 
		aSpace* newSpace = (aSpace*)(valloc(sizeof(aSpace))); 
		newSpace->space = stock_head;
		newSpace->prev = spaces;
		spaces = newSpace;
	}
	
	//get stock_step T's from the stock.
	//reuse space when possible, or move avail by stock_step.
	//if no more T's available, renew stock.
	//return pointer to T's
	T* fromMem() {
		T* temp;
		if(avail != stock_head){
			temp = avail;
			avail = (T*)((void**)(temp))[0];
		}
		else {
			if (stock_head == stock_tail)
				initMem();
			temp = stock_head;
			stock_head += stock_step;
			avail = stock_head;
		}
		return temp;
	}
	
	//return stock_step of T's to stock.
	// add to reusable space (avail)
	// and set incoming T* data to NIL;
	void toMem(T** addr) {
		assert(*addr != NIL);
		((void**)(*addr))[0] = (void*)(avail);
		avail = *addr;
		*addr = NIL;
	}

  //compute alloc_step (global)
	__attribute__ ((always_inline))
	void compute_alloc_steps(int baseBlockByteSize)
	{
		alloc_step = MEM_STEP / baseBlockByteSize;
	}

  //delete all allocated stocks and their storage
	void killMem() {
		avail = NIL;
		stock_head = NIL;
		stock_tail = NIL;
		aSpace *spacePtr = spaces, *tPtr;
		while (spacePtr != NIL) {
			tPtr = spacePtr;
			spacePtr = spacePtr->prev;
			if(tPtr->space != NIL)
        free(tPtr->space);
			tPtr->prev = NIL;
			free(tPtr);
		}
		spaces = NIL;
	}
	
	//default destructor
	~Memory(){}
};


#endif
//////////////////////////////////////////////////////////////////////////////



