/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

/*
 * This is testing features of the matrix class... to be replaced by actual unit tests in the future...
 */

#include <mpUtils/mpUtils.h>
//#include <mpUtils/mpGraphics.h>
#include <glm/glm.hpp>
#include <mpUtils/mpCuda.h>

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("matrixTest", "0.9.1");

    yield();

    logINFO("MtrixTesting") << "testing the matrix class..";
    logINFO("MtrixTesting") << "testing parameters..";

    Mat<float,4,6> m1; // 4 by 6 matrix with undefined values
    if(m1.size != 4*6)
        logERROR("MatrixTesting") << "size is wrong";
    if(m1.rows != 4)
        logERROR("MatrixTesting") << "rows wrong";
    if(m1.cols != 6)
        logERROR("MatrixTesting") << "cols wrong";

    logINFO("MatrixTesting") << "testing construction..";

    logINFO("MatrixTesting") << "constructing unit matrix";
    Mat<float,4,4> identitiy(1.0);
    myLog.print(LogLvl::INFO) << toString(identitiy);

    logINFO("MatrixTesting") << "constructing matrix with 5es in the diagonal";
    Mat<float,4,4> m2(5.0);
    myLog.print(LogLvl::INFO) << toString(m2);

    logINFO("MatrixTesting") << "constructing zero matrix";
    Mat<float,4,4> m0(0.0);
    myLog.print(LogLvl::INFO) << toString(m0);

    logINFO("MatrixTesting") << "constructing matrix as series of numbers from 1 to 18";
    Mat<float,6,3> m3( 1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                      10.0,11.0,12.0,
                      13.0,14.0,15.0,
                      16.0,17.0,18.0);
    myLog.print(LogLvl::INFO) << toString(m3);

    logINFO("MatrixTesting") << "constructing matrix from glm mat3";
    glm::mat3 glmmat(1,2,3,
                     4,5,6,
                     7,8,9);
    Mat<float,3,3> m4(glmmat);
    myLog.print(LogLvl::INFO) << toString(m4);

    logINFO("MatrixTesting") << "converting to glm mat3 and back";
    glm::mat3 glmmat3(m4);
    Mat<float,3,3> convert(glmmat3);
    myLog.print(LogLvl::INFO) << mpu::toString(convert);

    logINFO("MatrixTesting") << "testing component access..";
    logINFO("MatrixTesting") << "manually accessing data of last created matrix using double brackets:";
    myLog.print(LogLvl::INFO) << "[0][0]: "<<m4[0][0]<< " [0][1]: "<<m4[0][1]<< " [0][2]: "<<m4[0][2] <<"\n"
                              << "[1][0]: "<<m4[1][0]<< " [1][1]: "<<m4[1][1]<< " [1][2]: "<<m4[1][2] <<"\n"
                              << "[2][0]: "<<m4[2][0]<< " [2][1]: "<<m4[2][1]<< " [2][2]: "<<m4[2][2] <<"\n";

    logINFO("MatrixTesting") << "manually accessing data of last created matrix using round brackets:";
    myLog.print(LogLvl::INFO) << " 0: " << m4(0) << ", 1: " << m4(1) << ", 2: " << m4(2) << "\n"
                              << " 3: " << m4(3) << ", 4: " << m4(4) << ", 5: " << m4(5) << "\n"
                              << " 6: " << m4(6) << ", 7: " << m4(7) << ", 8: " << m4(8) << "\n";

    logINFO("MatrixTesting") << "modifing values:";
    m4[0][0] = 21;
    m4(1)= 42;
    myLog.print(LogLvl::INFO) << m4(0) << "   " << m4[0][1];


    logINFO("MatrixTesting") << "testing logical compare operators..";
    logINFO("MatrixTesting") << "expected true, false, false, true";
    Mat<float,4,4 > m5(1.0);
    myLog.print(LogLvl::INFO) << boolalpha << (m5==identitiy);
    myLog.print(LogLvl::INFO) << boolalpha << (m5==m0);
    myLog.print(LogLvl::INFO) << boolalpha << (m5!=identitiy);
    myLog.print(LogLvl::INFO) << boolalpha << (m5!=m0);


    logINFO("MatrixTesting") << "testing arithmetic operators..";
    logINFO("MatrixTesting") << "testing addition and subtraction operators..";
    logINFO("MatrixTesting") << "using the series of numbers matrix and the identity..";

    Mat<float,4,4> m6( 1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0,10.0,11.0,12.0,
                       13.0,14.0,15.0,16.0);

    logINFO("MatrixTesting") << "plus";
    myLog.print(LogLvl::INFO) << toString(identitiy + m6);

    logINFO("MatrixTesting") << "minus";
    myLog.print(LogLvl::INFO) << toString(identitiy - m6);

    logINFO("MatrixTesting") << "plus equal";
    auto r1=identitiy;
    r1 += m6;
    myLog.print(LogLvl::INFO) << toString(r1);

    logINFO("MatrixTesting") << "minus equal";
    auto r2=identitiy;
    r2 -= m6;
    myLog.print(LogLvl::INFO) << toString(r2);

    logINFO("MatrixTesting") << "testing scalar multiplication..";
    logINFO("MatrixTesting") << "using the series of numbers matrix..";

    logINFO("MatrixTesting") << "multiply by 2";
    myLog.print(LogLvl::INFO) << toString(m6*2);

    logINFO("MatrixTesting") << "multiply other way around";
    myLog.print(LogLvl::INFO) << toString(2.0f*m6);

    logINFO("MatrixTesting") << "divide by 2";
    myLog.print(LogLvl::INFO) << toString(m6/2);

    logINFO("MatrixTesting") << "multiply equal";
    auto r3=m6;
    r3*=2;
    myLog.print(LogLvl::INFO) << toString(r3);

    logINFO("MatrixTesting") << "divide equal";
    auto r4=m6;
    r4/=2;
    myLog.print(LogLvl::INFO) << toString(r4);

    logINFO("MatrixTesting") << "testing matrix multiplication..";
    logINFO("MatrixTesting") << "using the series of numbers matrix..";
    auto r5=m6;
    r5*=m6;
    r5*=identitiy;
    myLog.print(LogLvl::INFO) << toString(r5);

    Mat<float,4,2> m7(1,2,3,4,5,6,7,8);
    myLog.print(LogLvl::INFO) << toString(m6*m7);

    logINFO("MatrixTesting") << "testing matrix transpose..";
    myLog.print(LogLvl::INFO) << toString(transpose(m6));

    logINFO("MatrixTesting") << "testing component wise multiplication..";
    myLog.print(LogLvl::INFO) << toString(compWiseMult(m6, identitiy));

    logINFO("MatrixTesting") << "testing invert of 2x2..";
    Mat<float,2,2> m8( 0,5,
                       5,0);
    myLog.print(LogLvl::INFO) << toString(invert(m8));
    myLog.print(LogLvl::INFO) << toString(m8*invert(m8));
    myLog.print(LogLvl::INFO) << toString(invert(m8)*m8);

    Mat<float,3,3> m92( 0,0,5,
                        0,5,0,
                        5,0,0);
    logINFO("MatrixTesting") << "testing invert of 4x4..";
    myLog.print(LogLvl::INFO) << toString(invert(m92));
    myLog.print(LogLvl::INFO) << toString(m92*invert(m92));
    myLog.print(LogLvl::INFO) << toString(invert(m92)*m92);


    Mat<float,4,4> m9( 0,0,0,5,
                       0,0,5,0,
                       0,5,0,0,
                       5,0,0,0);
    logINFO("MatrixTesting") << "testing invert of 4x4..";
    myLog.print(LogLvl::INFO) << toString(invert(m9));
    myLog.print(LogLvl::INFO) << toString(m9*invert(m9));
    myLog.print(LogLvl::INFO) << toString(invert(m9)*m9);

    logINFO("MatrixTesting") << "testing matrix vector multiplication..";
    Mat<float,2,2> m10(2,4,3,5);
    Mat<float,3,3> m11(2,4,6,3,5,7,4,6,8);
    Mat<float,4,4> m12(2,4,6,7,
                       3,5,7,8,
                       4,6,8,9,
                       5,7,9,10);

    float2 f{3,5};
    f = m10*f;
    myLog.print(LogLvl::INFO) << f.x << ", " << f.y;

    float3 f2{3,5,7};
    f2 = m11*f2;
    myLog.print(LogLvl::INFO) << f2.x << ", " << f2.y << ", " << f2.z;

    float4 f3{3,5,7,9};
    f3 = m12*f3;
    myLog.print(LogLvl::INFO) << f3.x << ", " << f3.y << ", " << f3.z << ", " << f3.w;

    glm::vec2 g2(3,5);
    g2 = m10 * g2;
    myLog.print(LogLvl::INFO) << g2.x << ", " << g2.y;

    glm::vec3 g3(3,5,7);
    g3 = m11 * g3;
    myLog.print(LogLvl::INFO) << g3.x << ", " << g3.y << ", " << g3.z;

    glm::vec4 g4(3,5,7,9);
    g4 = m12 * g4;
    myLog.print(LogLvl::INFO) << g4.x << ", " << g4.y << ", " << g4.z << ", " << g4.w;

    return 0;
}