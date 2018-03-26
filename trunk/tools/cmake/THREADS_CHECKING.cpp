// Naming deviates from stardard convention to not be globbed and to give threads a special treatment

#include <thread>

void do_something() {}

int main()
{
    std::thread my_thread(do_something);
    my_thread.join();
    return 0;
}