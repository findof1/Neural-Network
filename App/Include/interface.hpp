#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QTextEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QThread>
#include <QTimer>
#include "neuralNetworkUtils.hpp"

class AppInterface : public QWidget
{
  Q_OBJECT
public:
  AppInterface(QWidget *parent = nullptr) : QWidget(parent)
  {
    setWindowTitle("Neural Network App");
    resize(1000, 700);

    auto *root = new QVBoxLayout(this);

    auto *title = new QLabel("Dashboard");
    title->setStyleSheet("font-size: 24px; font-weight: bold;");
    root->addWidget(title);

    auto *topRow = new QHBoxLayout();
    root->addLayout(topRow);

    auto *configBox = new QGroupBox("Training Config");
    auto *configLayout = new QVBoxLayout(configBox);

    epochs = new QSpinBox();
    epochs->setRange(1, 100000000);
    epochs->setValue(1000);

    batchSize = new QSpinBox();
    batchSize->setRange(1, 10000);
    batchSize->setValue(32);

    learningRate = new QDoubleSpinBox();
    learningRate->setDecimals(6);
    learningRate->setRange(0.000001, 10.0);
    learningRate->setValue(0.01);
    learningRate->setSingleStep(0.001);

    configLayout->addWidget(new QLabel("Epochs"));
    configLayout->addWidget(epochs);
    configLayout->addWidget(new QLabel("Batch Size"));
    configLayout->addWidget(batchSize);
    configLayout->addWidget(new QLabel("Learning Rate"));
    configLayout->addWidget(learningRate);

    startBtn = new QPushButton("Train Model");

    configLayout->addWidget(startBtn);
    configLayout->addStretch();

    topRow->addWidget(configBox, 1);

    // Todo: add live stats
    /*
    auto *statsBox = new QGroupBox("Live Stats");
    auto *statsLayout = new QVBoxLayout(statsBox);

    epochLabel = new QLabel("Epoch: 0");
    lossLabel = new QLabel("Loss: 0.0");
    accuracyLabel = new QLabel("Accuracy: 0.0%");
    progress = new QProgressBar();
    progress->setRange(0, 100);

    statsLayout->addWidget(epochLabel);
    statsLayout->addWidget(lossLabel);
    statsLayout->addWidget(accuracyLabel);
    statsLayout->addWidget(progress);
    statsLayout->addStretch();

    topRow->addWidget(statsBox, 2);

    log = new QTextEdit();
    log->setReadOnly(true);
    root->addWidget(log, 1);
    */

    connect(startBtn, &QPushButton::clicked, this, &AppInterface::startTraining);
  }

private slots:
  void startTraining()
  {
    // add live stats stuff here
    train();
  }

  void train()
  {
    Network net;
    net.config.epochs = epochs->value();
    net.config.learningRate = static_cast<float>(learningRate->value());
    net.config.batchSize = batchSize->value();
    // Digit Guesser:
    addLayer(net, 784, 256, ReLU);
    addLayer(net, 256, 128, ReLU);
    addLayer(net, 128, 64, ReLU);
    addLayer(net, 64, 10, SoftMax);

    // Example reconstruction network:
    // addLayer(net, 784, 128, ReLU);
    // addLayer(net, 128, 64, ReLU);
    // addLayer(net, 64, 128, ReLU);
    // addLayer(net, 128, 784, Sigmoid);

    std::cout << "Loading Training Dataset..." << std::endl;
    Dataset dataset = loadMNISTCSV("mnist_train.csv");

    std::cout << "Loading Test Dataset..." << std::endl;
    Dataset testDataset = loadMNISTCSV("mnist_test.csv");

    std::cout << "Beginning Training..." << std::endl;
    float totalLoss = 0.0f;
    std::mt19937 rng(std::random_device{}());
    int count = 0;
    for (int epoch = 0; epoch < net.config.epochs; epoch++)
    {
      std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);
      totalLoss = 0.0f;

      count = 0;

      for (auto &sample : dataset.samples)
      {
        forwardPass(net, sample.inputs);

        totalLoss += computeCostCrossEntropy(net, sample);

        backpropagation(net, sample);

        count++;

        if (count % net.config.batchSize == 0)
        {
          applyGradients(net);
        }
      }

      if (epoch % net.config.epochDisplayInterval == 0)
      {
        std::cout << "Epoch " << epoch << " Loss: " << totalLoss / dataset.samples.size() << std::endl;
      }
    }
    std::cout << "\nFinal Output:" << std::endl;
    std::cout << "Total Epochs " << net.config.epochs << " Loss: " << totalLoss / dataset.samples.size() << std::endl;
    std::cout << "Accuracy:" << computeAccuracy(net, testDataset) << std::endl;
  }

private:
  QPushButton *startBtn;
  // QLabel *epochLabel;
  // QLabel *lossLabel;
  // QLabel *accuracyLabel;
  // QTextEdit *log;
  // QProgressBar *progress;
  QSpinBox *epochs;
  QSpinBox *batchSize;
  QDoubleSpinBox *learningRate;
};
