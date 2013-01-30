// Filename: qr_givens_example.cpp (part of MTL4)

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;
typedef mtl::matrix::dense2D<double> dMatrix;

int main() {
    dMatrix M1(3,3), M2(3,3);
    

    M1 = 2,0,0,
	1,1,0,
	0,1,3; //EWs: 1,2,3     
	
    mtl::matrix::qr_givens<dMatrix> QR1(M1);
    QR1.setTolerance(1.0e-5);
    QR1.calc();
    cout << "M1(providing tolerance):\n Q: \n"  << QR1.getQ() << "\n R: \n" << QR1.getR() << "\n";      
    M2 = -261, 209,  -49,
        -530, 422,  -98,
        -800, 631, -144; //EWs: 3,4,10  
        
    mtl::matrix::qr_givens<dMatrix> QR2(M2);
    QR2.calc();
    cout << "M2(with defaults):\n Q: \n"  << QR1.getQ() << "\n R: \n" << QR1.getR() << "\n";
    
    return 0;
}
