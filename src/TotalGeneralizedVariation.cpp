//
// Created by roundedglint585 on 3/11/19.
//

#include "TotalGeneralizedVariation.hpp"
#include <fstream>

#define Debug

TotalGeneralizedVariation::TotalGeneralizedVariation(const std::vector<TotalGeneralizedVariation::Image> &images)
        : m_images(images), m_result(m_images[0]), m_width(m_result[0].size()), m_height(m_result.size()) {
    initGradientAndEpsilon();
    initWs();
    initStacked();
}

TotalGeneralizedVariation::TotalGeneralizedVariation(std::vector<TotalGeneralizedVariation::Image> &&images) : m_images(
        std::move(images)), m_result(m_images[0]), m_width(m_result[0].size()), m_height(m_result.size()) {
    initGradientAndEpsilon();
    initWs();
    initStacked();
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
            i = std::vector<float>(m_width, 0.0f);
        }
    }
}

void TotalGeneralizedVariation::initStacked() {
    stacked = std::vector(m_height,
                          std::vector<std::vector<float>>(m_width, std::vector<float>(Ws.size() + m_images.size(), 0)));
}


void TotalGeneralizedVariation::calculateHist() {
    for (size_t histNum = 0; histNum < Ws.size(); histNum++) {
        for (auto &image: m_images) {
            for (size_t i = 0; i < m_height; i++) {
                for (size_t j = 0; j < m_width; j++) {
                    if (m_images[histNum][i][j] > image[i][j]) {
                        Ws[histNum][i][j] += 1.f;
                    } else {
                        if (m_images[histNum][i][j] < image[i][j])
                            Ws[histNum][i][j] -= 1.f;
                    }
                }
            }
        }
    }

}


TotalGeneralizedVariation::Image
TotalGeneralizedVariation::prox(const TotalGeneralizedVariation::Image &image, float tau, float lambda_data) {
    Image result = Image(m_height, std::vector<float>(m_width, 0));
    //need to swap dimensions to calculate median
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            for (size_t k = 0; k < m_images.size(); k++) {
                stacked[i][j][k] = m_images[k][i][j];
            }
            for (size_t k = 0; k < Ws.size(); k++) {
                stacked[i][j][m_images.size() + k] = image[i][j] + tau * lambda_data * Ws[k][i][j];
            }
        }
    }

    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            std::stable_sort(stacked[i][j].begin(), stacked[i][j].end());

            if (stacked[i][j].size() % 2 == 1) {
                result[i][j] = stacked[i][j][stacked[i][j].size() / 2 + 1];
            } else {
                result[i][j] =
                        (stacked[i][j][stacked[i][j].size() / 2] + stacked[i][j][stacked[i][j].size() / 2 - 1]) / 2;
            }
        }
    }
    return result;
}


TotalGeneralizedVariation::Image
TotalGeneralizedVariation::solve(float tau, float lambda_tv, float lambda_tgv, float lambda_data, size_t iterations) {
    calculateHist();
    Image u = m_result;
    Gradient v = mathRoutine::calculateGradient(m_result);
    Gradient p = mathRoutine::calculateGradient(m_result);
    Epsilon q = mathRoutine::calculateEpsilon(v);

    for (size_t i = 0; i < iterations; i++) {
        if (i % 100 == 0) { std::cout << "Iteration #: " << i << std::endl; }

        tgvIteration(u, v, p, q, tau, lambda_tv, lambda_tgv, lambda_data);
    }
    return u;
}


void TotalGeneralizedVariation::tgvIteration(Image &u, Gradient &v, Gradient &p, Epsilon &q, float tau, float lambda_tv,
                                             float lambda_tgv, float lambda_data) {
    using namespace mathRoutine;
    float tau_u, tau_v, tau_p, tau_q;
    tau_u = tau;
    tau_v = tau;
    tau_p = tau;
    tau_q = tau;
    Image un = prox(u + (-tau_u * lambda_tv) * (mathRoutine::calculateTranspondedGradient(p)), tau_u, lambda_data);

    Gradient vn = v + (-lambda_tgv * tau_v) * mathRoutine::calculateTranspondedEpsilon(q) + (tau_v * lambda_tv) * p;

    Gradient pn = project(
            p + (tau_p * lambda_tv) * (mathRoutine::calculateGradient((-1) * ((-2) * un + u)) + ((-2) * vn + v)),
            lambda_tv);

    Epsilon qn = project(q + (-tau_q * lambda_tgv) * mathRoutine::calculateEpsilon(-2 * vn + v), lambda_tgv);
    u = std::move(un);
    v = std::move(vn);
    p = std::move(pn);
    q = std::move(qn);


}
