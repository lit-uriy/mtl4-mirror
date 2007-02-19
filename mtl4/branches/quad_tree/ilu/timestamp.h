/*****************************************************************************
  file: timestamp.h
  ----------------- 
  Defines function to starting and end timings
  Includes user, system and wall times

  Revised on: 07/25/06
  
  Larisse D. Voufo
*****************************************************************************/

#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>


//time stamp global variables
static int user_time, system_time;
static double wall_time;
static struct tms start_cpu, end_cpu;
static struct timeval start_wall, end_wall;
static int clock_ticks = sysconf(_SC_CLK_TCK);

/* 
void timestamp_start(char* title)
  Starts timing...
  param title:  indicating message, mostly for debugging purposes 
*/ 
inline void timestamp_start(char* title)
{
	printf("\n\t%s\n", title);
   	gettimeofday(&start_wall,NULL);    // Get starting time
	times(&start_cpu);
}

/* 
void timestamp_end(char* title)
  Ends timing and prints out times
  param title:  indicating message 
*/ 
inline void timestamp_end(char* title)
{
	times(&end_cpu);                // Get ending time
	gettimeofday(&end_wall,NULL);

  //compute times
	user_time = (end_cpu.tms_utime - start_cpu.tms_utime);
	system_time = (end_cpu.tms_stime - start_cpu.tms_stime);
	wall_time = ((end_wall.tv_sec - start_wall.tv_sec)
						+(1.0e-6)*(end_wall.tv_usec-start_wall.tv_usec));

  //print times
	printf("\n\t%s\n", title);
	printf(  "\t-----------------------------------\n");
	printf("\t Wall Time (seconds) %e\n", wall_time);
	printf("\t User Time (seconds) %e, System Time %e\n",
				(double)user_time/clock_ticks, (double)system_time/clock_ticks);
}

#endif
//////////////////////////////////////////////////////////////////////////////

