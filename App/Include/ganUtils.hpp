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