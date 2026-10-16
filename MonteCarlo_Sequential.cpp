/*
 * ARTI 503 - Stage 1
 * Sequential Monte Carlo simulation for one-year Value at Risk (VaR).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

const double INITIAL_INVESTMENT = 10000.0;
const double EXPECTED_RETURN = 0.08;
const double VOLATILITY = 0.20;
const double CONFIDENCE_LEVEL = 0.95;
const unsigned int SEED = 10;
const int REPETITIONS = 3;
const int SIZES[] = {10000, 100000, 1000000};

struct Results {
    double meanFinalValue;
    double meanReturnPercent;
    double meanProfitOrLoss;
    double var95;
    double probabilityOfLossPercent;
    double executionTimeSeconds;
};

Results runSimulation(int n) {
    // The timer covers simulation, sorting, and calculation of the outputs.
    auto start = std::chrono::steady_clock::now();

    std::mt19937 generator(SEED);
    std::normal_distribution<double> returnDistribution(
        EXPECTED_RETURN, VOLATILITY
    );

    std::vector<double> losses;
    losses.reserve(n);

    double totalLoss = 0.0;
    int lossCount = 0;

    for (int i = 0; i < n; ++i) {
        // A portfolio cannot lose more than its initial value.
        double annualReturn = std::max(-1.0, returnDistribution(generator));
        double finalValue = INITIAL_INVESTMENT * (1.0 + annualReturn);
        double profitOrLoss = finalValue - INITIAL_INVESTMENT;
        double loss = -profitOrLoss;

        losses.push_back(loss);
        totalLoss += loss;
        if (loss > 0.0) {
            ++lossCount;
        }
    }

    std::sort(losses.begin(), losses.end());

    // Nearest-rank method: index = ceil(0.95 * n) - 1.
    int rank = static_cast<int>(std::ceil(CONFIDENCE_LEVEL * n));
    double var95 = losses[rank - 1];
    double meanLoss = totalLoss / n;

    auto end = std::chrono::steady_clock::now();
    double executionTime = std::chrono::duration<double>(end - start).count();

    Results results;
    results.meanFinalValue = INITIAL_INVESTMENT - meanLoss;
    results.meanReturnPercent = (-meanLoss / INITIAL_INVESTMENT) * 100.0;
    results.meanProfitOrLoss = -meanLoss;
    results.var95 = var95;
    results.probabilityOfLossPercent = (static_cast<double>(lossCount) / n) * 100.0;
    results.executionTimeSeconds = executionTime;
    return results;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sequential Monte Carlo Simulation - Annual VaR\n";
    std::cout << "Initial investment: " << INITIAL_INVESTMENT << " SAR\n";
    std::cout << "Expected annual return: " << EXPECTED_RETURN * 100 << "%\n";
    std::cout << "Annual volatility: " << VOLATILITY * 100 << "%\n";
    std::cout << "Confidence level: " << CONFIDENCE_LEVEL * 100 << "%\n";
    std::cout << "Fixed seed: " << SEED << " | Repetitions: " << REPETITIONS << "\n";
    std::cout << "Note: returns below -100% are limited to -100%.\n";

    for (int n : SIZES) {
        double totalTime = 0.0;
        std::cout << "\nProblem size N = " << n << "\n";

        for (int repetition = 1; repetition <= REPETITIONS; ++repetition) {
            Results r = runSimulation(n);
            totalTime += r.executionTimeSeconds;

            std::cout << "Run " << repetition
                      << " | Mean final value: " << r.meanFinalValue
                      << " | Mean return: " << r.meanReturnPercent << "%"
                      << " | Mean profit/loss: " << r.meanProfitOrLoss
                      << " | VaR 95%: " << r.var95
                      << " | Probability of loss: " << r.probabilityOfLossPercent << "%"
                      << " | Time: " << r.executionTimeSeconds << " s\n";
        }

        std::cout << "Average execution time: " << totalTime / REPETITIONS << " s\n";
    }

    return 0;
}
