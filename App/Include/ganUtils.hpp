#pragma once
#include "neuralNetworkUtils.hpp"

struct GANNetwork
{
  Network generator;
  Network discriminator;
};

void discriminatorBackpropagation(GANNetwork &fullNetwork, Sample &sample);

void generatorBackpropagation(GANNetwork &fullNetwork, const Eigen::VectorXf &noise, const Eigen::VectorXf &discriminatorGradient);

float computeDiscriminatorLoss(const Eigen::VectorXf &realOut, const Eigen::VectorXf &fakeOut);

float computeGeneratorLoss(const Eigen::VectorXf &fakeOut);

Eigen::VectorXf getInputGradient(GANNetwork &fullNetwork, Sample &sample);

Eigen::VectorXf computeMeanImage(const Dataset &data, int digit);
Eigen::VectorXf computeStdDevImage(const Dataset &data, const Eigen::VectorXf &meanImage, int digit);
Eigen::VectorXf computeDatasetVariance(Dataset &data);
float getLatentRatio(GANNetwork &fullNetwork, const Eigen::VectorXf &digitInput, std::mt19937 &rng);