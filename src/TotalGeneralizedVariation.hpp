//
// Created by roundedglint585 on 3/11/19.
//

#ifndef TGV_TOTALGENERALIZEDVARATION_HPP
#define TGV_TOTALGENERALIZEDVARATION_HPP

#include <vector>
#include <array>
#include <cmath>
#include "MathRoutine.hpp"


constexpr float eps = 0.00001;

class TotalGeneralizedVariation {
public:

    using Image = std::vector<std::vector<float>>;
    using Gradient = std::vector<std::vector<std::array<float, 2>>>;
    using Epsilon = std::vector<std::vector<std::array<float, 4>>>;

    explicit TotalGeneralizedVariation(const std::vector<Image> &);

    explicit TotalGeneralizedVariation(std::vector<Image> &&);


public:
    //math routine

    void calculateGradient();


    void calculateEpsilon();


    void calculateTranspondedGradient();


    void calculateTranspondedEpsilon();


    void calculateHist();


    Image prox(const Image &image, float tau, float lambda_data) const;

    //
    Image solve(float tau, float lambda_tv, float lambda_tgv, float lambda_data, size_t iterations = 1000);

    void tgvIteration(Image &u, Gradient &v, Gradient &p, Epsilon q, float tau, float lambda_tv, float lambda_tgv,
                      float lambda_data);

    //
    void initGradientAndEpsilon();

    void initWs();

    std::vector<Image> m_images;
    std::vector<Image> Ws; //histogram fot images
    Image m_result;
    size_t m_width;
    size_t m_height;
    Gradient m_gradient;
    Image m_transpondedGradient;
    Epsilon m_epsilon;
    Gradient m_transpondedEpsilon;


};


#endif //TGV_TOTALGENERALIZEDVARATION_HPP
