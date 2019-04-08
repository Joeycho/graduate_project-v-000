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

      console.log("shuffled data before to tensor:", data);

      // Step 2. Convert data to Tensor
    //  const inputs = data.map(d => d.horsepower)
      const inputs = data.map(d => d.slice(0,numShops));

      const labels = data.map( d => {
        temparray = new Array(numShops).fill(0);
        temparray[d[numShops]]=1;

        return temparray;
      });

//      console.log("inputTensor: ",inputTensor);
      //const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
          console.log("inputs: ",inputs);
          console.log("labels: ",labels);



      const inputTensor = tf.tensor2d(inputs, [data.length,numShops]);
      const labelTensor = tf.tensor2d(labels, [labels.length, numShops]);

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

  function createModel(numShops) {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [numShops], units: 3, useBias: true}));

  // Second layer
  model.add(tf.layers.dense({units: 2, activation: 'relu'}));

  // Add an output layer
  model.add(tf.layers.dense({units: 5, useBias: true}));

  return model;
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 50;
  const epochs = 50;

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

function testModel(model, inputData, normalizationData, numShops, numDatas) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const totalNum = numShops*numDatas;

  const xsi = tf.linspace(0,1,totalNum);
  console.log("Before shuffle xsi: ",xsi);
  const xsarray = xsi.dataSync();
  tf.util.shuffle(xsarray);
  console.log("After shuffle, xsarray: ",xsarray);
  const xst = tf.tensor1d(xsarray);
  console.log("From xsarray to xst tensor: ",xst);

  xaxis = tf.linspace(1,numDatas,numDatas).dataSync();
  console.log("xaxis x:",Array.from(xaxis));


  const [xs, preds] = tf.tidy(() => {

        const preds = model.predict(xst.reshape([numDatas, numShops]));

    const unNormXs = xst
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

  const predictedPoints = Array.from(xaxis).map((val, i) => {
    return {x: val, y: Math.floor(preds[i])}
  });

  let maxIndex = 0;

  let original = [];

  for (let i = 0; i < totalNum; i++) {

      const seed = xsarray[i];
      if(seed>xsarray[maxIndex]){
          maxIndex = i;
      };

      if(i%numShops == numShops-1){
        original.push(maxIndex%numShops);
        maxIndex = i+1;
      };
  };

  console.log("orginal y:",original);


  const originalPoints = Array.from(xaxis).map((d,i) => {
    return{x: d, y: original[i]}
  });

  console.log("predicted: ",predictedPoints);

  console.log("original: ",originalPoints);


  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: `rating from ${numShops} shops`,
      yLabel: 'choice',
      height: 300
    }
  );
}


async function run() {

    const numShops = 5;
    const numDatas = 300;
    const data = generateData(10,numShops,numDatas);

    console.log(data);

    const examplesDiv = document.getElementById('dataExamples');
    for (let i = 0; i< numDatas;i++){
        examplesDiv.innerHTML += `${i+1} element in data: ${data[i]}`+'<br>';
    }

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data,numShops);
    const {inputs, labels} = tensorData;


    const model = createModel(numShops);
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model, data, tensorData,numShops, numDatas);
  }

  run();
