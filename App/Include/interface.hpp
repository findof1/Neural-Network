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
#include "trainer.hpp"
#include "ganTrainer.hpp"

class AppInterface : public QWidget
{
  Q_OBJECT
public:
  Network trainedNet;
  bool modelReady = false;
  GANNetwork trainedGAN;
  bool ganModelReady = false;

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

    auto *realtimeRow = new QHBoxLayout();
    root->addLayout(realtimeRow);
    auto *realtimeBox = new QGroupBox("Realtime Testing");
    auto *realtimeLayout = new QVBoxLayout(realtimeBox);

    testBtn = new QPushButton("Test Random Sample");
    realtimeLayout->addWidget(testBtn);

    inputImageLabel = new QLabel();
    inputImageLabel->setFixedSize(280, 280);
    inputImageLabel->setAlignment(Qt::AlignCenter);
    inputImageLabel->setStyleSheet("border: 1px solid gray;");

    realtimeLayout->addWidget(inputImageLabel, 1);

    outputImageLabel = new QLabel();
    outputImageLabel->setFixedSize(280, 280);
    outputImageLabel->setAlignment(Qt::AlignCenter);
    outputImageLabel->setStyleSheet("border: 1px solid gray;");

    realtimeLayout->addWidget(outputImageLabel, 2);

    realtimeRow->addWidget(realtimeBox);

    connect(startBtn, &QPushButton::clicked, this, &AppInterface::startTrainingGAN);

    connect(testBtn, &QPushButton::clicked, this, &AppInterface::testRandomSample);
  }

  void showSampleImage(const Eigen::VectorXf &vec, bool input)
  {
    const int w = 28;
    const int h = 28;

    QImage img(w, h, QImage::Format_Grayscale8);

    for (int y = 0; y < h; y++)
    {
      for (int x = 0; x < w; x++)
      {
        int index = y * w + x;

        float v = vec(index);

        if (v < 0.0f)
          v = 0.0f;
        if (v > 1.0f)
          v = 1.0f;

        int pixel = static_cast<int>(v * 255.0f);

        img.setPixel(x, y, pixel);
      }
    }

    if (input)
    {
      inputImageLabel->setPixmap(
          QPixmap::fromImage(img).scaled(
              inputImageLabel->size(),
              Qt::KeepAspectRatio,
              Qt::FastTransformation));
    }
    else
    {
      outputImageLabel->setPixmap(
          QPixmap::fromImage(img).scaled(
              outputImageLabel->size(),
              Qt::KeepAspectRatio,
              Qt::FastTransformation));
    }
  }

private slots:
  void startTrainingGAN()
  {
    thread = new QThread(this);
    ganTrainer = new GANTrainer();
    ganModelReady = false;

    ganTrainer->moveToThread(thread);

    connect(thread, &QThread::started, ganTrainer, [this]()
            { ganTrainer->run(
                  epochs->value(),
                  batchSize->value(),
                  (float)learningRate->value()); });

    connect(ganTrainer, &GANTrainer::statsUpdated,
            this, &AppInterface::updateStats,
            Qt::QueuedConnection);

    connect(ganTrainer, &GANTrainer::showSampleImage,
            this, &AppInterface::showSampleImage,
            Qt::QueuedConnection);

    connect(ganTrainer, &GANTrainer::finished,
            this, &AppInterface::ganTrainingFinished);

    connect(ganTrainer, &GANTrainer::printLog,
            this, &AppInterface::printLog,
            Qt::QueuedConnection);

    connect(ganTrainer, &GANTrainer::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, ganTrainer, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    thread->start();
  }

  void startTraining()
  {
    thread = new QThread(this);
    trainer = new Trainer();
    modelReady = false;

    trainer->moveToThread(thread);

    connect(thread, &QThread::started, trainer, [this]()
            { trainer->run(
                  epochs->value(),
                  batchSize->value(),
                  (float)learningRate->value()); });

    connect(trainer, &Trainer::statsUpdated,
            this, &AppInterface::updateStats,
            Qt::QueuedConnection);

    connect(trainer, &Trainer::showSampleImage,
            this, &AppInterface::showSampleImage,
            Qt::QueuedConnection);

    connect(trainer, &Trainer::finished,
            this, &AppInterface::trainingFinished);

    connect(trainer, &Trainer::printLog,
            this, &AppInterface::printLog,
            Qt::QueuedConnection);

    connect(trainer, &Trainer::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, trainer, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    thread->start();
  }

  void testRandomSample()
  {
    /*
    if (!modelReady)
    {
      log->append("Error: Cannot test sample on invalid model");
      return;
    }

    emit printLog("Loading Test Dataset");
    QString base = QCoreApplication::applicationDirPath();
    Dataset testDataset = loadMNISTCSV((base + "/mnist_test/mnist_test.csv").toStdString());
    std::mt19937 rng(std::random_device{}());

    int index = std::uniform_int_distribution<int>(0, testDataset.samples.size() - 1)(rng);

    Sample &sample = testDataset.samples[index];
    showSampleImage(sample.targets, true);
    forwardPass(trainedNet, sample.inputs);
    showSampleImage(trainedNet.layers.back().a, false);*/

    if (!ganModelReady)
    {
      log->append("Error: Cannot test sample on invalid gan model");
      return;
    }
    Eigen::VectorXf z(100);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < 100; i++)
      z(i) = dist(rng);
    forwardPass(trainedGAN.generator, z);
    showSampleImage(trainedGAN.generator.layers.back().a, true);
  }

  void trainingFinished()
  {
    log->append("Training Complete");

    trainedNet = trainer->net;
    modelReady = true;
  }

  void ganTrainingFinished()
  {
    log->append("Training Complete");

    trainedGAN = ganTrainer->net;
    ganModelReady = true;
  }

  void updateStats(int epoch, float loss, float acc, int percent)
  {
    epochLabel->setText("Epoch: " + QString::number(epoch));
    lossLabel->setText("Loss: " + QString::number(loss));
    accuracyLabel->setText("Accuracy: " + QString::number(acc * 100.0f) + "%");
    progress->setValue(percent);

    log->append("Epoch " + QString::number(epoch) +
                " Loss: " + QString::number(loss));
  }

  void printLog(const QString &str)
  {
    log->append(str);
  }

private:
  QPushButton *startBtn;
  QPushButton *testBtn;
  QLabel *epochLabel;
  QLabel *lossLabel;
  QLabel *accuracyLabel;
  QTextEdit *log;
  QProgressBar *progress;
  QSpinBox *epochs;
  QSpinBox *batchSize;
  QDoubleSpinBox *learningRate;
  QLabel *inputImageLabel;
  QLabel *outputImageLabel;

  QThread *thread;
  Trainer *trainer;
  GANTrainer *ganTrainer;
};
