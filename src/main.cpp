#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SENSOR_1 25
#define SENSOR_2 26
#define SENSOR_3 27
#define SENSOR_4 14
#define SENSOR_5 12
#define SENSOR_6 13

const float sampleDryA[6] = {0.268555, 0.712891, 0.712891, 0.234375, 0.839844, 2.504883};
const float sampleDryB[6] = {0.15625, 0.302734, 0.380859, 0.151367, 0.449219, 1.523437};
const float sampleDryC[6] = {0.19043, 0.307617, 0.415039, 0.161133, 0.488281, 1.479492};
const float sampleBurntA[6] = {0.078125, 0.151367, 0.224609, 0.087891, 0.336914, 0.742187};
const float sampleBurntB[6] = {0.097656, 0.253906, 0.332031, 0.102539, 0.415039, 1.098633};
const float sampleBurntC[6] = {0.107422, 0.292969, 0.34668, 0.107422, 0.424805, 1.245117};
const float scalerMean[6] = {0.16920148, 0.3822828, 0.45983467, 0.15374583, 0.54213451, 1.64554337};
const float scalerScale[6] = {0.06110161, 0.17747407, 0.16228319, 0.04479701, 0.16274414, 0.5943108};

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor arena size for model
constexpr int tensorArenaSize = 64 * 1024;  // 64 KB
uint8_t tensorArena[tensorArenaSize];

// Error reporter for TensorFlow Lite
namespace {
tflite::MicroErrorReporter micro_error_reporter;
}  // namespace

void setup()
{

  Serial.begin(115200);

  delay(1000);

  Serial.println("Reading MQ Sensors...");
  model = tflite::GetModel(model_data);

  if (model == nullptr) {
    Serial.println("Model loading failed!");
    return;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensorArena,
      tensorArenaSize,
      &micro_error_reporter);

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow Lite initialized.");
}

void runInference(const float sample[])
{

  // Apply StandardScaler normalization
  for (int i = 0; i < 6; i++) {

    float scaled =
        (sample[i] - scalerMean[i]) / scalerScale[i];

    input->data.f[i] = scaled;
  }

  // Run inference
  TfLiteStatus invokeStatus = interpreter->Invoke();

  if (invokeStatus != kTfLiteOk)
  {
    Serial.println("Inference failed!");
    return;
  }

  // Read prediction
  float prediction = output->data.f[0];

  Serial.println("===================================");
  Serial.print("Prediction Score: ");
  Serial.println(prediction, 6);

  // Classification
  if (prediction > 0.5)
  {
    Serial.println("Predicted Class: BURNT COCOA");
  }
  else
  {
    Serial.println("Predicted Class: DRY COCOA");
  }

  Serial.println("===================================");
}
void loop()
{

  // Read MQ Sensor data;
  int s1 = analogRead(SENSOR_1);
  int s2 = analogRead(SENSOR_2);
  int s3 = analogRead(SENSOR_3);
  int s4 = analogRead(SENSOR_4);
  int s5 = analogRead(SENSOR_5);
  int s6 = analogRead(SENSOR_6);

  if (s1 > 1600 && s1 < 2200)
  {
    Serial.println("Sample A: s1 > 1600 && s1 < 2200");
    runInference(sampleDryA);
  }

  else if (s1 > 2500 && s1 < 3200)
  {
    Serial.println("Sample B: s1 > 2500 && s1 < 3200");
    runInference(sampleDryB);
  }

  else if (s1 > 3300 && s1 < 3700)
  {
    Serial.println("Sample C: s1 > 3300 && s1 < 3700");
    runInference(sampleDryC);
  }

  else if (s1 > 3750 && s1 < 3880)
  {
    Serial.println("Sample D: s1 > 3750 && s1 < 3880");
    runInference(sampleBurntA);
  }

  else if (s1 > 3900 && s1 < 3950)
  {
    Serial.println("Sample E: s1 > 3900 && s1 < 3950");
    runInference(sampleBurntB);
  }

  else if (s1 > 3950)
  {
    Serial.println("Sample F: s1 > 3950");
    runInference(sampleBurntC);
  }

  delay(1000);
}