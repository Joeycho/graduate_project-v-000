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

async function runGenerateData() {

    const numDatas = 10;
    const data = generateData(10,5,numDatas);
    
    console.log(data);

    const examplesDiv = document.getElementById('dataExamples');
    for (let i = 0; i< numDatas;i++){
        examplesDiv.innerHTML += `${i+1} element in data: ${data[i]}`+'<br>';
    }
    
  }
  
  runGenerateData();