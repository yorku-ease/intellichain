const express = require("express");
const exec = require("child_process").exec;

const app = express();

app.use(express.json());

app.post("/ftm-predict", (req, res) => {
  const {input} = req.body;

  exec(
    `python scripts/ftm/predict_ftmscan.py ${input.join(" ")}`,
    (err, stdout, stderr) => {
      if (err) {
        console.log(err);
      }

      res.json(stdout);
    }
  );
});

app.post("/polygon-predict", (req, res) => {
  const {input} = req.body;

  exec(
    `python scripts/poly/predict_polygonscan.py ${input.join(" ")}`,
    (err, stdout, stderr) => {
      if (err) {
        console.log(err);
      }

      res.json(stdout);
    }
  );
});

app.listen(3000, (req, res) => {
  console.log("Server is running at port 3000.");
});
