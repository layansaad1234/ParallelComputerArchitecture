/*
 * ARTI 503 - Stage 1
 * Sequential code for the annual VaR simulation.
 *
 * The simulation generates annual returns using a normal distribution.
 * It then calculates the final investment value and the loss for each run.
 * VaR is taken as the 95th percentile of the simulated losses.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr double INITIAL_INVESTMENT = 10000.0;
constexpr double EXPECTED_ANNUAL_RETURN = 0.08;
constexpr double ANNUAL_VOLATILITY = 0.20;
constexpr double CONFIDENCE_LEVEL = 0.95;
constexpr unsigned int RANDOM_SEED = 10;
constexpr int REPETITIONS = 3;

const std::vector<std::size_t> PROBLEM_SIZES = {
    10000, 100000, 1000000
};

struct SimulationResult {
    std::size_t n;
    double annualVaR;
    double probabilityOfLossPercent;
    double meanLoss;
    double minLoss;
    double maxLoss;
};

struct BenchmarkRecord {
    std::size_t n;
    int repetition;
    unsigned int seed;
    double executionTimeSeconds;
    SimulationResult result;
};

SimulationResult simulateAnnualVaR(std::size_t n, unsigned int seed) {
    if (n == 0) {
        throw std::invalid_argument("The number of scenarios must be positive.");
    }

    std::mt19937 generator(seed);
    std::normal_distribution<double> annualReturnDistribution(
        EXPECTED_ANNUAL_RETURN, ANNUAL_VOLATILITY
    );

    // Store each simulated loss so we can sort them and calculate VaR.
    std::vector<double> losses;
    losses.reserve(n);

    std::size_t lossCount = 0;
    double lossSum = 0.0;
    double minLoss = 0.0;
    double maxLoss = 0.0;

    // Run the simulation once for each scenario.
    for (std::size_t i = 0; i < n; ++i) {
        const double annualReturn =
    std::max(-1.0, annualReturnDistribution(generator));

    const double finalValue =
    INITIAL_INVESTMENT * (1.0 + annualReturn);

    const double loss =
    INITIAL_INVESTMENT - finalValue;

        losses.push_back(loss);
        lossSum += loss;

        if (loss > 0.0) {
            ++lossCount;
        }

        if (i == 0) {
            minLoss = loss;
            maxLoss = loss;
        } else {
            if (loss < minLoss) minLoss = loss;
            if (loss > maxLoss) maxLoss = loss;
        }
    }

    // Sorting the losses lets us select the required percentile.
    std::sort(losses.begin(), losses.end());

    // Find the position for the 95th percentile in the sorted losses.
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(CONFIDENCE_LEVEL * static_cast<double>(n))
    );
    const std::size_t index = (rank == 0) ? 0 : rank - 1;

    return {
        n,
        losses[index],
        100.0 * static_cast<double>(lossCount) / static_cast<double>(n),
        lossSum / static_cast<double>(n),
        minLoss,
        maxLoss
    };
}

int main() {
    std::vector<BenchmarkRecord> records;
    records.reserve(PROBLEM_SIZES.size() * REPETITIONS);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sequential Monte Carlo Simulation - Annual VaR\n";
    std::cout << "Expected annual return: " << EXPECTED_ANNUAL_RETURN * 100.0 << "%\n";
    std::cout << "Annual volatility: " << ANNUAL_VOLATILITY * 100.0 << "%\n";
    std::cout << "Confidence level: " << CONFIDENCE_LEVEL * 100.0 << "%\n";
    std::cout << "Initial investment: " << INITIAL_INVESTMENT
              << " (currency/unit to be specified by the team)\n";
    std::cout << "Random seed: " << RANDOM_SEED << "\n";

    // Test the three problem sizes and repeat each one three times.
    for (const std::size_t n : PROBLEM_SIZES) {
        std::cout << "\nProblem size N = " << n << '\n';
        double totalTime = 0.0;

        for (int repetition = 1; repetition <= REPETITIONS; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            const SimulationResult result = simulateAnnualVaR(n, RANDOM_SEED);
            const auto end = std::chrono::steady_clock::now();

            const std::chrono::duration<double> elapsed = end - start;
            totalTime += elapsed.count();

            records.push_back({
                n, repetition, RANDOM_SEED, elapsed.count(), result
            });

            std::cout
                << "  Run " << repetition
                << " | Time: " << elapsed.count() << " s"
                << " | Annual VaR (95%): " << result.annualVaR
                << " | Probability of loss: "
                << result.probabilityOfLossPercent << "%\n";
        }

        std::cout << "  Average execution time: "
                  << totalTime / static_cast<double>(REPETITIONS)
                  << " s\n";
    }

    // Save the results so they can be used in the report and tables.
    const std::string outputFile = "sequential_benchmark_results.csv";
    std::ofstream csv(outputFile);

    if (!csv) {
        std::cerr << "Error: Could not create " << outputFile << '\n';
        return 1;
    }

    csv << "N,repetition,seed,execution_time_seconds,annual_var_95,"
           "probability_of_loss_percent,mean_loss,min_loss,max_loss\n";
    csv << std::fixed << std::setprecision(10);

    for (const auto& record : records) {
        csv << record.n << ','
            << record.repetition << ','
            << record.seed << ','
            << record.executionTimeSeconds << ','
            << record.result.annualVaR << ','
            << record.result.probabilityOfLossPercent << ','
            << record.result.meanLoss << ','
            << record.result.minLoss << ','
            << record.result.maxLoss << '\n';
    }

    csv.close();
    std::cout << "\nDetailed results saved to " << outputFile << '\n';
    std::cout
        << "\nModeling note: simulated returns below -100% are "
        << "floored at -100% to prevent negative final investment values.\n";

    return 0;
}
