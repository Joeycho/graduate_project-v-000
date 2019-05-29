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

function generateData(ratingRange, numShops, numElements) {

    //Generate random numbers in the given range, and make a element,
    //which contains them in the number of Shops and ends with the index (0 ~ numShops-1),which has highest number.
    //Add elements as much as numElements counts.
    //One data consists of several elements and each element consists of generated numbers and index.

    const output = [];

    for (let k = 0; k < numElements; k++){
    const element = [];
        let maxIndex = 0;
        for (let i = 0; i < numShops; i++) {
            const seed = Math.round(Math.random() * ratingRange*100)/100;

            element.push(seed);

            if(seed>element[maxIndex]){
                maxIndex = i;
            }
        };

        element.push(maxIndex);
        output.push(element);
    }
    return output;
  }

  function convertToTensor(data,numShops) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      //Shuffle the data
      tf.util.shuffle(data);


      //Convert data to Tensor
      const inputs = data.map(d => d.slice(0,numShops));

      const labels = data.map( d => {
        temparray = new Array(numShops).fill(0);
        temparray[d[numShops]]=1;
        return temparray;
      });

      const inputTensor = tf.tensor2d(inputs, [data.length,numShops]);
      const labelTensor = tf.tensor2d(labels, [labels.length, numShops]);

      //Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        //Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  function createModel(numShops) {
  //Create a sequential model
  const model = tf.sequential();

  //Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [numShops], units: 20, useBias: true}));

  //Add second layer
  model.add(tf.layers.dense({units: 18, activation: 'sigmoid', useBias: true}));

  //Add an output layer
  model.add(tf.layers.dense({units: numShops, activation: 'sigmoid'}));

  return model;
}

async function trainModel(model, inputs, labels, batchS, epocs) {
  //Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = batchS;
  const epochs = epocs;
 
  const lossContainer = document.getElementById('lossChart');

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
//     { name: 'Training Performance' },

    callbacks: tfvis.show.fitCallbacks(
      lossContainer,
      ['loss', 'mse'],
      { height: 300, callbacks: ['onEpochEnd'] }
    )
  });
}

async function testModel(model, inputData, normalizationData, numShops, numElements) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  //Generate predictions for a uniform range of numbers between 0 and 1;
  //We un-normalize the data by doing the inverse of the min-max scaling

  const totalNum = numShops*numElements;

  const xsi = tf.linspace(0,1,totalNum);
  console.log("Before shuffle xsi: ",xsi);
  const xsarray = xsi.dataSync();
  tf.util.shuffle(xsarray);
  console.log("After shuffle, xsarray: ",xsarray);
  const xst = tf.tensor1d(xsarray);
  console.log("From xsarray to xst tensor: ",xst.print());

  xaxis = tf.linspace(1,numElements,numElements).dataSync();
  console.log("xaxis x:",Array.from(xaxis));


  const [xs, preds] = tf.tidy(() => {

        const preds = model.predict(xst.reshape([numElements, numShops]));
        console.log("Before change to unNormpreds: ",preds.print());

    const unNormXs = xst
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

      console.log("After change to unNormpreds: ",unNormPreds);

    //Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.arraySync()];
  });

  console.log("xs: ",xs);
  console.log("preds: ",preds);

  let maxIndexp = 0;

  let originalp = [];

  for (let k = 0; k < numElements; k++) {

    for(let i = 0; i<numShops;i++){
      const seed = preds[k][i];
      if(seed>preds[k][maxIndexp]){
          maxIndexp = i;
      };

      if(i == numShops-1){
        originalp.push(maxIndexp);
        maxIndexp = 0;
      };
    }

  };

  const predictedPoints = Array.from(xaxis).map((val, i) => {
    return {x: val, y: originalp[i]}
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

  let ratio = 0;
  for(let i=0;i<numElements;i++){
    if(predictedPoints[i]['y']==originalPoints[i]['y']){
      ratio++;
    };
  }
  ratio = ratio/numElements*100;

  console.log("Hit ratio:",ratio);

  const hitratioDiv = document.getElementById('hitratio');

  hitratioDiv.innerHTML =`HIT Ratio:${ratio}%`+`<br>`;

  const compareContainer = document.getElementById('compareResult');
 //    {name: 'Model Predictions vs Original Data', tab: 'Evaluation'},
 tfvis.render.scatterplot(
    compareContainer, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: `rating from ${numShops} shops`,
      yLabel: 'choice',
      height: 300
    }
  );

  const confusionMatrix = await tfvis.metrics.confusionMatrix(tf.tensor1d(original),tf.tensor1d(originalp));

//  const container = {name: 'confusionMatrix', tab: 'Evaluation'};
  const confusionContainer = document.getElementById('confusionContainer');

  const choices = tf.linspace(1,numShops,numShops).arraySync();

  tfvis.render.confusionMatrix(confusionContainer, {values: confusionMatrix, tickLabels: Array.from(choices)});
}


async function run() {
 
 document.getElementById('trainModel').addEventListener('click', async () => {
   
    const batchsize = +(document.getElementById('batchsize')).value;
    const epochs = +(document.getElementById('epochs')).value;
    const numofshops = +(document.getElementById('numofshops')).value;
    const numofelements = +(document.getElementById('numofelements')).value;
    const rangeofrating = +(document.getElementById('rangeofrating')).value;

/*    // Do some checks on the user-specified parameters.
    const status = document.getElementById('trainStatus');
    if (digits < 1 || digits > 5) {
      status.textContent = 'digits must be >= 1 and <= 5';
      return;
    }
    const trainingSizeLimit = Math.pow(Math.pow(10, digits), 2);
    if (trainingSize > trainingSizeLimit) {
      status.textContent =
          `With digits = ${digits}, you cannot have more than ` +
          `${trainingSizeLimit} examples`;
      return;
    }*/

   const data = generateData(rangeofrating,numofshops,numofelements);

    //Convert the data to a tensor form we can use for training.
    const tensorData = convertToTensor(data,numofshops);
    const {inputs, labels} = tensorData;

    //Create model to train
    const model = createModel(numofshops);
    //Print the status of models
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    //Train the model
    await trainModel(model, inputs, labels,batchsize, epochs);
    console.log('Done Training');

    //Test the model
    testModel(model, data, tensorData,numofshops, numofelements);

  });
/*    const numShops = 5;
    const numElements = 500;
    const range = 100;

    const data = generateData(range,numShops,numElements);

    //Print subtitle in the page
    const subtitleDiv = document.getElementById('subtitle');
    subtitleDiv.innerHTML +=`Data should be array which contains ${numShops} randomly generated ratings and one choice`;

    //Print each element in the generated data
    const examplesDiv = document.getElementById('dataExamples');
    for (let i = 0; i< numElements;i++){
        examplesDiv.innerHTML += `${i+1} element in data: ${data[i]}`+'<br>';
    }

    //Convert the data to a tensor form we can use for training.
    const tensorData = convertToTensor(data,numShops);
    const {inputs, labels} = tensorData;

    //Create model to train
    const model = createModel(numShops);
    //Print the status of models
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    //Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    //Test the model
    testModel(model, data, tensorData,numShops, numElements);*/

  }

  run();
