//
// Created by roundedglint585 on 3/11/19.
//

#include "TotalGeneralizedVariation.hpp"
#include <algorithm>

TotalGeneralizedVariation::TotalGeneralizedVariation(const std::vector<TotalGeneralizedVariation::Image> &images)
        : m_images(images), m_result(m_images[0]), m_width(m_result[0].size()), m_height(m_result.size()) {
    initGradientAndEpsilon();
    initWs();
}

TotalGeneralizedVariation::TotalGeneralizedVariation(std::vector<TotalGeneralizedVariation::Image> &&images) : m_images(
        std::move(images)), m_result(m_images[0]), m_width(m_result[0].size()), m_height(m_result.size()) {
    initGradientAndEpsilon();
    initWs();
}

void TotalGeneralizedVariation::initGradientAndEpsilon() {
    m_gradient.resize(m_height);
    m_epsilon.resize(m_height);
    m_transpondedGradient.resize(m_height);
    m_transpondedEpsilon.resize(m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_gradient[i].resize(m_width);
        m_epsilon[i].resize(m_width);
        m_transpondedGradient[i].resize(m_width);
        m_transpondedEpsilon[i].resize(m_width);
    }
}

void TotalGeneralizedVariation::initWs() {
    Ws.resize(m_images.size());
    for (auto &ws: Ws) {
        ws.resize(m_height);
        for (auto &i : ws) {
            i = std::vector(m_width, 0.0f);
        }
    }
}

void TotalGeneralizedVariation::calculateGradient() {
    m_gradient = mathRoutine::calculateGradient(m_result);
}

void TotalGeneralizedVariation::calculateEpsilon() {
    m_epsilon = mathRoutine::calculateEpsilon(m_gradient);
}

void TotalGeneralizedVariation::calculateTranspondedGradient() {
    m_transpondedGradient = mathRoutine::calculateTranspondedGradient(m_gradient);
}

void TotalGeneralizedVariation::calculateTranspondedEpsilon() {
    m_transpondedEpsilon = mathRoutine::calculateTranspondedEpsilon(m_epsilon);
}

void TotalGeneralizedVariation::calculateHist() {
    for (size_t histNum = 0; histNum < Ws.size(); histNum++) {
        for (auto &image: m_images) {
            for (size_t i = 0; i < m_height; i++) {
                for (size_t j = 0; j < m_width; j++) {
                    if (m_images[histNum][i][j] > image[i][j]) {
                        Ws[histNum][i][j]++;
                    } else {
                        Ws[histNum][i][j]--;
                    }
                }
            }
        }
    }
}


TotalGeneralizedVariation::Image
TotalGeneralizedVariation::prox(const TotalGeneralizedVariation::Image &image, float tau, float lambda_data) const {
    Image result = Image(m_height, std::vector<float>(m_width, 0));
    std::vector<std::vector<std::vector<float>>> stacked; //need to swap dimensions to calculate median
    stacked.resize(m_height);
    for (size_t i = 0; i < m_height; i++) {
        stacked[i].resize(m_width);
        for (size_t j = 0; j < m_width; j++) {
            stacked[i][j] = std::vector(m_images.size() + Ws.size(), 0.0f);
            for (size_t k = 0; k < m_images.size(); k++) {
                stacked[i][j][k] = image[i][j];
            }
            for (size_t k = 0; k < Ws.size(); k++) {
                stacked[i][j][k] = image[i][j] + tau * lambda_data * Ws[i][j][k];
            }
        }
    }
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            std::nth_element(stacked[i][j].begin(), stacked[i][j].begin() + stacked[i][j].size() / 2 + 1,
                             stacked[i][j].end());
            if (Ws.size() % 2 == 1) {
                result[i][j] = stacked[i][j][Ws.size() / 2 + 1];
            } else {
                result[i][j] = (stacked[i][j][Ws.size() / 2 + 1] + stacked[i][j][Ws.size() / 2]) / 2;
            }
        }
    }
    return result;
}


TotalGeneralizedVariation::Image
TotalGeneralizedVariation::solve(float tau, float lambda_tv, float lambda_tgv, float lambda_data, size_t iterations) {
    calculateGradient();
    calculateEpsilon();
    calculateTranspondedGradient();
    calculateTranspondedEpsilon();
    calculateHist();
    Image u = m_result;
    Gradient v = m_gradient;
    Gradient p = m_gradient;
    Epsilon q = m_epsilon;
    for (size_t i = 0; i < iterations; i++) {
        tgvIteration(u, v, p, q, tau, lambda_tv, lambda_tgv, lambda_data);
    }
    return u;
}


void TotalGeneralizedVariation::tgvIteration(Image &u, Gradient &v, Gradient &p, Epsilon q, float tau, float lambda_tv,
                                             float lambda_tgv, float lambda_data) {
    using namespace mathRoutine;
    float tau_u, tau_v, tau_p, tau_q;
    tau_u = tau;
    tau_v = tau;
    tau_p = tau;
    tau_q = tau;
    Image un = prox(u + (-1) * tau_u * lambda_tv * (mathRoutine::calculateTranspondedGradient(p)), tau_u, lambda_data);
    Gradient vn = v + tau_v * (lambda_tgv * ((-1) * (mathRoutine::calculateTranspondedEpsilon(q))) + lambda_tv * p);
    Gradient pn = project(
            p + tau_p * lambda_tv * (mathRoutine::calculateGradient(2 * un + (-1) * u) + (-1) * (2 * vn + (-1) * v)),
            lambda_tv);
    Epsilon qn = project(q + tau_q * lambda_tgv * mathRoutine::calculateEpsilon(2 * vn + (-1) * v), lambda_tgv);
    u = std::move(un);
    v = std::move(vn);
    p = std::move(pn);
    q = std::move(qn);

}
