#pragma once
#include <QCoreApplication>
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

class Trainer : public QObject
{
  Q_OBJECT

public slots:
  void run(int epochs, int batchSize, float lr)
  {
    Network net;
    net.config.epochs = epochs;
    net.config.batchSize = batchSize;
    net.config.learningRate = lr;

    addLayer(net, 784, 256, ReLU);
    addLayer(net, 256, 128, ReLU);
    addLayer(net, 128, 64, ReLU);
    addLayer(net, 64, 10, SoftMax);

    emit printLog("Loading Training Dataset");
    QString base = QCoreApplication::applicationDirPath();
    Dataset dataset = loadMNISTCSV((base + "/mnist_train/mnist_train.csv").toStdString());
    emit printLog("Loading Test Dataset");
    Dataset testDataset = loadMNISTCSV((base + "/mnist_test/mnist_test.csv").toStdString());
    emit printLog("Beginning Training");

    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < net.config.epochs; epoch++)
    {
      std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);

      float totalLoss = 0.0f;
      int count = 0;

      for (auto &sample : dataset.samples)
      {
        forwardPass(net, sample.inputs);
        totalLoss += computeCostCrossEntropy(net, sample);
        backpropagation(net, sample);

        count++;

        if (count % net.config.batchSize == 0)
          applyGradients(net);
      }

      if ((epoch + 1) % net.config.epochDisplayInterval == 0)
      {
        float loss = totalLoss / dataset.samples.size();
        float acc = computeAccuracy(net, testDataset);
        int progress = (epoch + 1) * 100 / net.config.epochs;

        emit statsUpdated(epoch + 1, loss, acc, progress);
      }
    }

    emit finished();
  }

signals:
  void statsUpdated(int epoch, float loss, float acc, int progress);
  void finished();
  void printLog(const QString &str);
};