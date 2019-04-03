/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

function generateData(ratingRange, numShops, numRow) {

    const output = [];

    for (let k = 0; k < numRow; k++){
    const row = [];
        let maxIndex = 0;
        for (let i = 0; i < numShops; i++) {
            const seed = Math.round(Math.random() * ratingRange*100)/100;

            row.push(seed);

            if(seed>row[maxIndex]){
                maxIndex = i;
            }
        };

        row.push(maxIndex);
        output.push(row);
    }
    return output;
  }

  function convertToTensor(data,numShops) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);

      // Step 2. Convert data to Tensor
    //  const inputs = data.map(d => d.horsepower)
      const inputs = data.map(d => d.slice(0,numShops))
      const labels = data.map(d => d[numShops]);


//      console.log("inputTensor: ",inputTensor);
      //const labelTensor = tf.tensor2d(labels, [labels.length, 1]);


      const inputTensor = tf.tensor2d(inputs, [data.length, numShops]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [5], units: 1, useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 14;
  const epochs = 30;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 300, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xs = tf.linspace(0,1,100);
    console.log("Before shuffle xs: ",xs);
    const xsarray = xs.dataSync();
    tf.util.shuffle(xsarray);
    console.log("After shuffle, xsarray: ",xsarray);
    const preds = model.predict(xs.reshape([20, 5]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  console.log("xs: ",xs);
  console.log("preds: ",preds);

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.slice(0,5), y: d[5],
  }));


  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'rating from 5 shops',
      yLabel: 'choice',
      height: 300
    }
  );
}


async function run() {

    const numShops = 5;
    const numDatas = 20;
    const data = generateData(10,numShops,numDatas);

    console.log(data);

    const examplesDiv = document.getElementById('dataExamples');
    for (let i = 0; i< numDatas;i++){
        examplesDiv.innerHTML += `${i+1} element in data: ${data[i]}`+'<br>';
    }

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data,numShops);
    const {inputs, labels} = tensorData;

    console.log("inputs: ",inputs);
    console.log("labels: ",labels);

    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model, data, tensorData);
  }

  run();
