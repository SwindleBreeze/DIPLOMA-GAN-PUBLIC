class GroundGenerationHandler {
    constructor() {
        this.safeGroundPerGeneration = {}; // {safeCoordinates: [generation, generation, generation]}, if generations reach 3 then generate with conditionHigh
        this.unsafeGroundPerGeneration = {};  // {unsafeCoordinates: [generation, generation, generation]}, if generations reach 3 then generate with conditionLow
        this.conditionLow = 8;
        this.conditionHigh = 14;
        this.usPoI = {x: 0, y: 0};
        this.sPoI = {x: 0, y: 0};

        this.prevSPoI = {x: 0, y: 0};
        this.prevUSPoI = {x: 0, y: 0};
    }

    addGround(generation, coordinates, isSafe) {
        // console.log(generation, coordinates, isSafe);
        // console.log(coordinates.x, coordinates.y);

        if (isSafe) {
            if (this.safeGroundPerGeneration[coordinates.x] === undefined) {
                this.safeGroundPerGeneration[coordinates.x] = [generation];
            } else {
                this.safeGroundPerGeneration[coordinates.x].push(generation);
            }
        } else {
            if (this.unsafeGroundPerGeneration[coordinates.x] === undefined) {
                this.unsafeGroundPerGeneration[coordinates.x] = [generation];
            } else {
                this.unsafeGroundPerGeneration[coordinates.x].push(generation);
            }
        }
    }

    clearValues() {
        this.safeGroundPerGeneration = {};
        this.unsafeGroundPerGeneration = {};

        this.prevSPoI = {x: this.sPoI.x, y: this.sPoI.y};
        this.prevUSPoI = {x: this.usPoI.x, y: this.usPoI.y};

        this.sPoI = {x: 0, y: 0};
        this.usPoI = {x: 0, y: 0};
    }

    incrementConditionLow(delta) {
        this.conditionLow += delta;
    }

    incrementConditionHigh(delta) {
        this.conditionHigh += delta;
    }

    calculateCoordinatesFromDeathData(deathData) {
        for (let generation in deathData) {
            let coordinates = deathData[generation];
            for (let coordinate of coordinates) {
                let x = Math.round(coordinate.x);
                let y = Math.round(coordinate.y);
                this.addGround(generation, {x: x, y: y}, false);
            }
        }

        // get max x from deathData
        let maxX = -500;
        let keys = Object.keys(this.unsafeGroundPerGeneration);

        for (let key of keys) {
            if (parseInt(key) > maxX) {
                maxX = parseInt(key);
            }
        }


        console.log("MAX X IS: " + maxX);

        for (let i = 0; i < maxX; i++) {
            if (this.unsafeGroundPerGeneration[i] === undefined || this.unsafeGroundPerGeneration[i].length < 1) {
                for (let generation in deathData) {
                    this.addGround(generation, {x: i, y: 0}, true);
                }
            }
        }
    }

    calculatePointsOfInterest() {
        // find the coordinate that appears the most in the unsafeGroundPerGeneration
        let max = 0;
        
        let unsafePointOfInterest = 0;
        for (let coordinate in this.unsafeGroundPerGeneration) {
            let count = this.unsafeGroundPerGeneration[coordinate].length;
            if (count > max) {
                max = count;
                unsafePointOfInterest = coordinate;
            }
        }

        this.usPoI = {x: unsafePointOfInterest, y: 0};

        // find the coordinate that appears the most in the safeGroundPerGeneration, if there are multiple coordinates with the same count, choose one randomly
        max = 0;
        let safePointOfInterest = 0;
        let possibleSPoIs = [];

        for (let coordinate in this.safeGroundPerGeneration) {
            let count = this.safeGroundPerGeneration[coordinate].length;
            if (count == max) {
                possibleSPoIs.push(coordinate);
            } else if (count > max) {
                max = count;
            }
        }

        if (possibleSPoIs.length > 0) {
            let randomIndex = Math.floor(Math.random() * possibleSPoIs.length);
            safePointOfInterest = possibleSPoIs[randomIndex];
        } else {
            safePointOfInterest = coordinate;
        }

        this.sPoI = {x: safePointOfInterest, y: 0};

        console.log("All possible safe points of interest: ", possibleSPoIs);
        console.log("Safe Point of Interest: ["+ this.sPoI.x + ", 0]");
        console.log("Unsafe Point of Interest: [", this.usPoI.x + ", 0]");

    }
}

// basic express server

const express = require('express');
var cors = require('cors');
var fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');

let port = 3000;
let app = express();

// enable parsing of json data
app.use(express.json({limit: '250mb'}));

//enable cors
app.use(cors());


let groundGenerationHandler = new GroundGenerationHandler();
// Function to write JSON data to a file
const writeToFile = (filename, data) => {
    fs.writeFileSync(filename, JSON.stringify(data), 'utf8');
};
app.post('/ground', (req, res) => {

    // check how many files are in ../learning-sdata
    fs.readdir('../learning-data', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            // console.log("I contain this many files: " + files.length)
            let newFileName = `../learning-data/ground-${files.length + 1 + uuidv4()}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    });

    res.send('Data received');
});

app.post('/finalground', (req, res) => {
    fs.readdir('../validation-data', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../validation-data/gen-ground-${files.length + 1 - 36}-edited.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    });
});

app.post('/generationreguned', (req, res) => {
    fs.readdir('../gen-data/reguned', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../gen-data/reguned/generation-${files.length + 1}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    });
    res.send('Data received');
});

app.get('/flatground', (req, res) => {
    let id = req.query.id
    let fileName = `../validation-data/perlin-ground-${id}.json`;
    fs.readFile(fileName, (err, data) => {
        if (err) {
            console.log(err);
        } else {
            // convert buffer to js object
            let tmp = JSON.parse(data);
            res.send([tmp]);
        }
    });
});

app.post('/generationgenuned', (req, res) => {
    fs.readdir('../gen-data/genuned', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../gen-data/genuned/generation-${files.length + 1}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    });
    res.send('Data received');
});

app.post('/generationflat', (req, res) => {
    fs.readdir('../gen-data/perlin', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../gen-data/perlin/generation-${files.length + 1}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            }
            );
        }
    }
    );
    res.send('Data received');
});


app.post('/generationreged', (req, res) => {
    fs.readdir('../gen-data/reged', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../gen-data/reged/generation-${files.length + 1}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    });
    res.send('Data received');
});

app.post('/generationgened', (req, res) => {
    console.log(req.body);
    fs.readdir('../gen-data/gened', (err, files) => {
        if (err) {
            console.log(err);
        } else {
            let newFileName = `../gen-data/gened/generation-${files.length + 1}.json`;
            console.log(newFileName);
            fs.writeFileSync(newFileName, JSON.stringify(req.body), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('File saved');
                }
            });
        }
    }
    );
    res.send('Data received');
});

app.post('/test', (req, res) => {
    console.log(req.body);
    res.send('Data received');
});

app.get('/ground', (req, res) => {
    console.log(req)
    let id = req.query.id
    let fileName = `../learning-data/ground-${id}.json`;
    fs.readFile(fileName, (err, data) => {
        if (err) {
            console.log(err);
        } else {
            res.send(data);
        }
    });
});

app.get('/genground', (req, res) => {
    let id = req.query.id
    let fileName = `../validation-data/gen-ground-${id}.json`;
    fs.readFile(fileName, (err, data) => {
        if (err) {
            console.log(err);
        } else {
            let tmp = JSON.parse(data);
            res.send([tmp]);
        }
    });
});

app.get('/randground', (req, res) => {
    let id = req.query.id
    let fileName = `../validation-data/rand-ground-${id}.json`;
    fs.readFile(fileName, (err, data) => {
        if (err) {
            console.log(err);
        } else {
            // convert buffer to js object
            let tmp = JSON.parse(data);
            res.send([tmp]);
        }
    });
});

app.post('/modifyground', (req, res) => {
    let id = req.body.id;
    let groundData = req.body.groundData;
    let deathData = req.body.generationDeathData

    groundGenerationHandler.calculateCoordinatesFromDeathData(deathData);
    groundGenerationHandler.calculatePointsOfInterest();
    // console.log(groundGenerationHandler);


    if (groundGenerationHandler.prevSPoI.x === groundGenerationHandler.sPoI.x) {
        groundGenerationHandler.incrementConditionHigh(0.5);
    }
    else groundGenerationHandler.conditionHigh = 6;

    if (groundGenerationHandler.prevUSPoI.x === groundGenerationHandler.usPoI.x) {
        groundGenerationHandler.incrementConditionLow(0.5);
    } else groundGenerationHandler.conditionLow = 2;

    writeToFile('tmpGroundData.json', groundData);

    console.log(groundGenerationHandler.conditionLow, groundGenerationHandler.conditionHigh);
    console.log(JSON.stringify(groundGenerationHandler.sPoI), JSON.stringify(groundGenerationHandler.usPoI))

    // pass js array to python script
    const pythonProcess = spawn('python', ['modify_ground.py', groundGenerationHandler.conditionLow, groundGenerationHandler.conditionHigh, 'tmpGroundData.json', JSON.stringify(groundGenerationHandler.sPoI), JSON.stringify(groundGenerationHandler.usPoI)]);
    let data = '';

    // Capture stdout data
    pythonProcess.stdout.on('data', (chunk) => {
        data += chunk.toString();
    });

    // Handle any errors
    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    // Send response when process is closed

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            res.status(500).send('Failed to modify ground data');
            return;
        }
        try {
            let modifiedData = JSON.parse(data);
            res.json(modifiedData);
        } catch (err) {
            res.status(500).send('Error parsing modified ground data');
        }
    });

    groundGenerationHandler.clearValues();
    groundGenerationHandler.incrementConditionLow(0);
    groundGenerationHandler.incrementConditionHigh(0);
});

app.get('/map', (req, res) => {
    const pythonProcess = spawn('python', ['generate_ground_gan.py']);

    let data = '';

    // Capture stdout data
    pythonProcess.stdout.on('data', (chunk) => {
        data += chunk.toString();
    });

    // Handle any errors
    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    // Send response when process is closed
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            res.status(500).send('Failed to generate map data');
            return;
        }
        try {
            const mapData = JSON.parse(data);
            res.json(mapData);
        } catch (err) {
            res.status(500).send('Error parsing map data');
        }
    });
});


app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});