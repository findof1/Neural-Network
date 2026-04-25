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
#include "ganUtils.hpp"

class GANTrainer : public QObject
{
  Q_OBJECT
public:
  GANNetwork net;

public slots:
  void run(int epochs, int batchSize, float lr)
  {
    net.generator.config.epochs = epochs;
    net.generator.config.batchSize = batchSize;
    net.generator.config.learningRate = lr;

    addLayer(net.generator, 110, 128, ReLU);
    addLayer(net.generator, 128, 256, ReLU);
    addLayer(net.generator, 256, 512, ReLU);
    addLayer(net.generator, 512, 784, Sigmoid);

    addLayer(net.discriminator, 784, 512, ReLU);
    addLayer(net.discriminator, 512, 256, ReLU);
    addLayer(net.discriminator, 256, 128, ReLU);
    addLayer(net.discriminator, 128, 2, Sigmoid);

    emit printLog("Loading Training Dataset");
    QString base = QCoreApplication::applicationDirPath();
    Dataset dataset = loadMNISTCSV((base + "/mnist_train/mnist_train.csv").toStdString());
    emit printLog("Beginning Training");

    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < net.generator.config.epochs; epoch++)
    {
      std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);

      float totalLossGen = 0.0f;
      float totalLossDis = 0.0f;
      int count = 0;

      for (auto &sample : dataset.samples)
      {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        Eigen::VectorXf z(100);
        for (int i = 0; i < 100; i++)
          z(i) = dist(rng);

        std::uniform_int_distribution<int> digitDist(0, 9);
        int c = digitDist(rng);
        Eigen::VectorXf oneHot(10);
        oneHot.setZero();
        oneHot(c) = 1.0f;

        Eigen::VectorXf input(110);
        input << z, oneHot;

        forwardPass(net.generator, input);
        Eigen::VectorXf fakeData = net.generator.layers.back().a;

        forwardPass(net.discriminator, sample.inputs);

        Sample realSample;
        realSample.inputs = sample.inputs;
        realSample.targets = Eigen::VectorXf(2);
        realSample.targets << 1.0f, 0.0f;

        discriminatorBackpropagation(net, realSample);

        forwardPass(net.discriminator, fakeData);

        Sample fakeSample;
        fakeSample.inputs = fakeData;
        fakeSample.targets = Eigen::VectorXf(2);
        fakeSample.targets << 0.0f, 1.0f;

        discriminatorBackpropagation(net, fakeSample);

        applyGradients(net.discriminator);

        Eigen::VectorXf realOut = net.discriminator.layers.back().a;

        forwardPass(net.generator, input);
        fakeData = net.generator.layers.back().a;

        forwardPass(net.discriminator, fakeData);

        Sample genTarget;
        genTarget.inputs = fakeData;
        genTarget.targets = Eigen::VectorXf(2);
        genTarget.targets << 1.0f, 0.0f;

        Eigen::VectorXf fakeGradient = getInputGradient(net, genTarget);

        generatorBackpropagation(net, input, fakeGradient);

        applyGradients(net.generator);

        Eigen::VectorXf fakeOut = net.discriminator.layers.back().a;
        totalLossDis += computeDiscriminatorLoss(realOut, fakeOut);
        totalLossGen += computeGeneratorLoss(fakeOut);

        count++;

        if (count % net.generator.config.batchSize == 0)
        {
          // add back if I fix batching
          //  applyGradients(net.generator);
          //  applyGradients(net.discriminator);
        }

        if (count % 6000 == 0)
        {
          std::cout << "6000 samples complete" << std::endl;
        }
      }

      if ((epoch + 1) % net.generator.config.epochDisplayInterval == 0)
      {
        float lossGen = totalLossGen / dataset.samples.size();
        float lossDis = totalLossDis / dataset.samples.size();
        int progress = (epoch + 1) * 100 / net.generator.config.epochs;

        emit statsUpdated(epoch + 1, lossGen, 0, progress);
        emit printLog(QString("Current Generator Loss: %1").arg(lossGen));
        emit printLog(QString("Current Discriminator Loss: %1").arg(lossDis));
      }
    }

    emit finished();
  }

signals:
  void statsUpdated(int epoch, float loss, float acc, int progress);
  void finished();
  void printLog(const QString &str);
  void showSampleImage(const Eigen::VectorXf &vec, bool input);
};