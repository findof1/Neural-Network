#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

#include <Eigen/Dense>
#include <QImage>

struct Sample
{
  Eigen::VectorXf inputs;
  Eigen::VectorXf targets;
};

struct Dataset
{
  std::vector<Sample> samples;
};

struct TrainingConfig
{
  float learningRate = 0.05f;
  int batchSize = 32;
  int epochs = 30;
  int epochDisplayInterval = 1;
};

enum LayerActivationType
{
  ReLU,
  Sigmoid,
  SoftMax,
  None
};

struct Layer
{
  Eigen::MatrixXf W;
  Eigen::VectorXf b;

  Eigen::MatrixXf dW;
  Eigen::VectorXf db;

  Eigen::VectorXf z;
  Eigen::VectorXf a;
  Eigen::VectorXf delta;

  LayerActivationType activation;
};

struct Network
{
  TrainingConfig config;
  std::vector<Layer> layers;
};

void forwardPass(Network &net, const Eigen::VectorXf &input);
float computeCostMSE(Network &net, const Sample &sample);
float computeCostCrossEntropy(Network &net, const Sample &sample);
void backpropagation(Network &net, const Sample &sample);
void applyGradients(Network &net);
void addLayer(Network &net, int inputSize, int outputSize, LayerActivationType activation);
Dataset loadMNISTCSV(const std::string &path);
int predict(Network &net, const Eigen::VectorXf &input);
float computeAccuracy(Network &net, const Dataset &dataset);