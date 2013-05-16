#include <initializer_list>

struct my_vector
{
    my_vector(std::initializer_list<int> ls) {}
};

int main()
{
    my_vector v= {3, 4, 5};
    const my_vector w= {3, 4, 5};

    return 0;
}
