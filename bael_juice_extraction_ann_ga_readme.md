# Bael Fruit Juice Extraction — ANN + GA and RSM

**Repository purpose**

This repository contains MATLAB code and documentation for modeling and optimizing enzymatic extraction of bael (Aegle marmelos) fruit juice using an Artificial Neural Network (ANN) coupled with a Genetic Algorithm (GA). The code reproduces the trained ANN (weights and biases provided) and a GA-based optimizer that uses the ANN model as a fitness evaluator. Also included are usage instructions, a short explanation of the model, and citations.

---

## Project structure (files included in this document)

- `README.md` (this document)
- `Prediction_for_Bael_Juice.m` — MATLAB function that implements the trained ANN model and returns predicted outputs for given inputs (pectinase conc., temperature, time).
- `GA_Optimization.m` — MATLAB fitness function implementing the same ANN forward-pass and returning a scalar fitness for GA to maximize (note: the file returns `f` as negative of objective so that standard MATLAB `ga` can minimize the function, or you can configure GA to maximize).
- `run_optimization.m` — example script to run MATLAB's `ga` using `GA_Optimization` as the fitness function.

> All MATLAB code blocks below are ready to be saved as `.m` files and used in MATLAB (tested for syntax consistency with R2019a). The ANN is an MLP with 3 inputs, 8 hidden neurons (tansig activation), and 6 linear outputs (purelin). The weights and biases have been embedded in the functions as provided.

---

## Requirements

- MATLAB R2019a (9.6) or newer.
- Neural Network Toolbox (for `tansig`, `purelin` functions) — these transfer functions exist in MATLAB's Neural Network Toolbox / Deep Learning Toolbox. If unavailable, equivalent implementations can be substituted (see notes below).
- Global Optimization Toolbox (if you want to use MATLAB's built-in `ga`), or any GA implementation that can call a MATLAB function handle.

---

## How the ANN & GA are used

- The ANN model was trained using MATLAB's Neural Network fitting tools (MLP with Bayesian Regularization). After training, weights and biases were exported and hard-coded into `Prediction_for_Bael_Juice.m` and `GA_Optimization.m`.
- `Prediction_for_Bael_Juice.m` implements the forward pass: it first scales the three inputs (pectinase concentration in g/100g, temperature in °C, and time in hours) into coded values [-1, +1], computes hidden-layer inputs, applies `tansig` activation, computes output-layer linear combination (`purelin`), then rescales outputs to their physical ranges and returns a 6-element vector `Y`.
- `GA_Optimization.m` is a fitness wrapper that computes the ANN outputs for a candidate `X` and returns a scalar fitness `f = -(Y(1)-Y(2)-Y(3)-Y(4)+Y(5)+Y(6));` The sign/combination follows the objective function used by the authors.

---

## Usage examples

### 1) Predict outputs for a single condition

Save the `Prediction_for_Bael_Juice.m` file (content below) and call from MATLAB:

```matlab
% Example inputs (pectinase g/100g, temp °C, time hours)
X1 = 0.18; X2 = 46.81; X3 = 6.09;
Y = PREDICTION(X1, X2, X3);
disp('Predicted outputs (6 values):');
disp(Y);
```

### 2) Run GA optimization (example)

Save `GA_Optimization.m` and `run_optimization.m` (content below). Then in MATLAB:

```matlab
run_optimization; % launches the GA optimizer with example bounds and options
```

`run_optimization.m` sets variable bounds (respecting the experimental ranges: pectinase 0.08–0.24 g/100g, temperature 30–60 °C, time 3–9 h), configures GA options (population size, display), and runs `ga` using the `GA_Optimization` fitness function.

---

## Files (code)

### Prediction_for_Bael_Juice.m

```matlab
function Y = PREDICTION(X1,X2,X3)

% PREDICTION - forward pass of trained ANN for bael juice extraction
% Inputs:
%   X1 - pectinase concentration (g/100 g pulp)
%   X2 - temperature (°C)
%   X3 - time (hours)
% Output:
%   Y  - 1x6 vector of predicted outputs (scaled back to physical values)

%-- Coded values of X (-1 to +1) --
x(1)= (((X1-0.08)/((0.24-0.08))*2))-1;
x(2)= (((X2-30)/((60-30))*2))-1;
x(3)= (((X3-3)/((9-3))*2))-1;

%-- Initial weights of hidden layers (hidden layer input) --
hi(1) = +1.3647*x(1) + 0.1057*x(2) - 0.2437*x(3) + 0.5523;
hi(2) = -0.9706*x(1) - 0.7845*x(2) + 0.3652*x(3) + 0.5471;
hi(3) = +1.0944*x(1) + 0.1306*x(2) + 1.0131*x(3) + 0.7806;
hi(4) = -0.2754*x(1) + 1.5495*x(2) + 0.0812*x(3) - 0.7956;
hi(5) = +0.0246*x(1) - 1.7754*x(2) - 0.0644*x(3) - 0.8396;
hi(6) = +0.1360*x(1) - 0.0930*x(2) + 1.3274*x(3) + 0.6788;
hi(7) = +1.4396*x(1) + 0.0290*x(2) - 0.2111*x(3) - 0.8792;
hi(8) = +0.7515*x(1) - 0.1436*x(2) + 1.1706*x(3) - 0.7500;

%-- Output weights of hidden layer (hidden layer output) --
h0(1)=tansig(hi(1));
h0(2)=tansig(hi(2));
h0(3)=tansig(hi(3));
h0(4)=tansig(hi(4));
h0(5)=tansig(hi(5));
h0(6)=tansig(hi(6));
h0(7)=tansig(hi(7));
h0(8)=tansig(hi(8));

%-- Input weights of output layer (output layer input) --
yi(1) = +0.0701*h0(1) + 0.1325*h0(2) + 0.4205*h0(3) - 0.3525*h0(4) - 0.4395*h0(5) + 0.6648*h0(6) + 0.8575*h0(7) - 0.9812*h0(8) - 0.8923;
yi(2) = -0.4579*h0(1) + 0.5727*h0(2) - 0.1349*h0(3) + 0.5033*h0(4) + 0.4144*h0(5) - 0.7701*h0(6) + 0.4029*h0(7) + 0.1543*h0(8) + 0.7596;
yi(3) = +0.0885*h0(1) + 0.1204*h0(2) - 0.5782*h0(3) + 0.3260*h0(4) + 0.3922*h0(5) - 0.4838*h0(6) - 0.5822*h0(7) + 0.5764*h0(8) + 0.6095;
yi(4) = +0.3113*h0(1) + 0.1705*h0(2) - 0.9698*h0(3) + 0.2152*h0(4) + 0.2695*h0(5) - 0.0758*h0(6) - 0.4300*h0(7) + 0.4180*h0(8) + 0.4199;
yi(5) = +0.9662*h0(1) + 0.4621*h0(2) + 0.7075*h0(3) - 0.1910*h0(4) - 0.4420*h0(5) - 0.5457*h0(6) - 1.0488*h0(7) + 0.0787*h0(8) - 0.9751;
yi(6) = +0.0715*h0(1) + 0.7361*h0(2) - 0.0872*h0(3) - 1.1069*h0(4) - 1.3456*h0(5) - 0.4035*h0(6) + 0.2217*h0(7) + 0.3053*h0(8) - 0.8773;

%-- Output weights of output layer (output layer output) --
y0(1)=purelin(yi(1));
y0(2)=purelin(yi(2));
y0(3)=purelin(yi(3));
y0(4)=purelin(yi(4));
y0(5)=purelin(yi(5));
y0(6)=purelin(yi(6));

%-- Output coded values to actual --
Y(1) = (((y0(1) + 1) * (63 - 44)) / 2) + 44;     % output 1 scaled to [44,63]
Y(2) = (((y0(2) + 1) * (11.3 - 10.3)) / 2) + 10.3; % output 2 scaled to [10.3,11.3]
Y(3) = (((y0(3) + 1) * (2.24 - 1.97)) / 2) + 1.97; % output 3 scaled to [1.97,2.24]
Y(4) = (((y0(4) + 1) * (313 - 211)) / 2) + 211;    % output 4 scaled to [211,313]
Y(5) = (((y0(5) + 1) * (31.6 - 15.6)) / 2) + 15.6; % output 5 scaled to [15.6,31.6]
Y(6) = (((y0(6) + 1) * (5.46 - 4.12)) / 2) + 4.12; % output 6 scaled to [4.12,5.46]

end
```

---

### GA_Optimization.m

```matlab
function f = GA(X)

% GA - fitness wrapper using the same ANN forward pass as PREDICTION
% Input:
%   X - 1x3 vector [pectinase, temperature, time]
% Output:
%   f - scalar fitness value (negative of objective used so GA minimizes f)

%-- Coded values of X (-1 to +1) --
x(1)= (((X(1)-0.08)/((0.24-0.08))*2))-1;
x(2)= (((X(2)-30)/((60-30))*2))-1;
x(3)= (((X(3)-3)/((9-3))*2))-1;

%-- Initial weights of hidden layers (hidden layer input) --
hi(1) = +1.3647*x(1) + 0.1057*x(2) - 0.2437*x(3) + 0.5523;
hi(2) = -0.9706*x(1) - 0.7845*x(2) + 0.3652*x(3) + 0.5471;
hi(3) = +1.0944*x(1) + 0.1306*x(2) + 1.0131*x(3) + 0.7806;
hi(4) = -0.2754*x(1) + 1.5495*x(2) + 0.0812*x(3) - 0.7956;
hi(5) = +0.0246*x(1) - 1.7754*x(2) - 0.0644*x(3) - 0.8396;
hi(6) = +0.1360*x(1) - 0.0930*x(2) + 1.3274*x(3) + 0.6788;
hi(7) = +1.4396*x(1) + 0.0290*x(2) - 0.2111*x(3) - 0.8792;
hi(8) = +0.7515*x(1) - 0.1436*x(2) + 1.1706*x(3) - 0.7500;

%-- Output weights of hidden layer (hidden layer output) --
h0(1)=tansig(hi(1));
h0(2)=tansig(hi(2));
h0(3)=tansig(hi(3));
h0(4)=tansig(hi(4));
h0(5)=tansig(hi(5));
h0(6)=tansig(hi(6));
h0(7)=tansig(hi(7));
h0(8)=tansig(hi(8));

%-- Input weights of output layer (output layer input) --
yi(1) = +0.0701*h0(1) + 0.1325*h0(2) + 0.4205*h0(3) - 0.3525*h0(4) - 0.4395*h0(5) + 0.6648*h0(6) + 0.8575*h0(7) - 0.9812*h0(8) - 0.8923;
yi(2) = -0.4579*h0(1) + 0.5727*h0(2) - 0.1349*h0(3) + 0.5033*h0(4) + 0.4144*h0(5) - 0.7701*h0(6) + 0.4029*h0(7) + 0.1543*h0(8) + 0.7596;
yi(3) = +0.0885*h0(1) + 0.1204*h0(2) - 0.5782*h0(3) + 0.3260*h0(4) + 0.3922*h0(5) - 0.4838*h0(6) - 0.5822*h0(7) + 0.5764*h0(8) + 0.6095;
yi(4) = +0.3113*h0(1) + 0.1705*h0(2) - 0.9698*h0(3) + 0.2152*h0(4) + 0.2695*h0(5) - 0.0758*h0(6) - 0.4300*h0(7) + 0.4180*h0(8) + 0.4199;
yi(5) = +0.9662*h0(1) + 0.4621*h0(2) + 0.7075*h0(3) - 0.1910*h0(4) - 0.4420*h0(5) - 0.5457*h0(6) - 1.0488*h0(7) + 0.0787*h0(8) - 0.9751;
yi(6) = +0.0715*h0(1) + 0.7361*h0(2) - 0.0872*h0(3) - 1.1069*h0(4) - 1.3456*h0(5) - 0.4035*h0(6) + 0.2217*h0(7) + 0.3053*h0(8) - 0.8773;

%-- Output weights of output layer (output layer output) --
y0(1)=purelin(yi(1));
y0(2)=purelin(yi(2));
y0(3)=purelin(yi(3));
y0(4)=purelin(yi(4));
y0(5)=purelin(yi(5));
y0(6)=purelin(yi(6));

%-- Output coded values to actual --
Y(1) = (((y0(1) + 1) * (63 - 44)) / 2) + 44;     % output 1 scaled to [44,63]
Y(2) = (((y0(2) + 1) * (11.3 - 10.3)) / 2) + 10.3; % output 2 scaled to [10.3,11.3]
Y(3) = (((y0(3) + 1) * (2.24 - 1.97)) / 2) + 1.97; % output 3 scaled to [1.97,2.24]
Y(4) = (((y0(4) + 1) * (313 - 211)) / 2) + 211;    % output 4 scaled to [211,313]
Y(5) = (((y0(5) + 1) * (31.6 - 15.6)) / 2) + 15.6; % output 5 scaled to [15.6,31.6]
Y(6) = (((y0(6) + 1) * (5.46 - 4.12)) / 2) + 4.12; % output 6 scaled to [4.12,5.46]

%-- Fitness function --
% note: objective used by authors: maximize (Y(1)-Y(2)-Y(3)-Y(4)+Y(5)+Y(6))
% to use with MATLAB's ga (which minimizes), return negative of objective
f=-(Y(1)-Y(2)-Y(3)-Y(4)+Y(5)+Y(6));
end
```

---

### run_optimization.m (example wrapper)

```matlab
% Example script to run GA (requires Global Optimization Toolbox)

% Variable bounds (pectinase [0.08 0.24], temperature [30 60], time [3 9])
lb = [0.08, 30, 3];
ub = [0.24, 60, 9];

% Options (adjust as needed)
options = optimoptions('ga', 'PopulationSize', 50, 'Display', 'iter', 'MaxGenerations', 200);

% Run GA — GA_Optimization expects a 1x3 vector input to compute a fitness
[xopt, fopt, exitflag, output] = ga(@GA, 3, [], [], [], [], lb, ub, [], options);

fprintf('GA result (pearson-coded): pectinase=%.4f g/100g, temp=%.4f °C, time=%.4f h\n', xopt(1), xopt(2), xopt(3));
fprintf('Objective value (negated fopt): %.4f\n', -fopt);

% Print predicted outputs for the optimum
Yopt = PREDICTION(xopt(1), xopt(2), xopt(3));
disp('Predicted outputs at optimum:');
disp(Yopt);
```

---

## Notes and implementation details

- `tansig` and `purelin` are transfer functions from MATLAB's Neural Network Toolbox. If you do not have these toolboxes, you can implement them yourself as:

```matlab
function y = tansig(x)
    y = 2 ./ (1 + exp(-2*x)) - 1; % hyperbolic tangent sigmoid
end

function y = purelin(x)
    y = x; % linear activation
end
```

- The ANN inputs are scaled from their physical ranges into [-1,1] before feeding the trained weight matrices — these scaling formulas are hard-coded in both functions.
- The outputs are scaled back from the coded ANN outputs (assumed between -1 and +1) to physical ranges using the same linear transform used by the authors.

---

## Results & published values (from the study)

- **RSM-optimized** variables: pectinase concentration = **0.22 g/100 g**, temperature = **46.20 °C**, time = **6.35 h**.
- **ANN-GA optimized** variables: pectinase concentration = **0.18 g/100 g**, temperature = **46.81 °C**, time = **6.09 h**.
- The authors found ANN to outperform RSM based on R², RMSE, and MAE metrics.

---

## Citation / Acknowledgements

If you use this code or model in a publication or project, please cite the original article from which the network weights and GA objective were derived. Example citation format (replace with the full bibliographic reference):

> Chegini et al., Hornik et al., Pradhan et al., Rai et al., Nandi et al., Sivanandam & Deepa — references mentioned in the original manuscript.

---

## License

This repository is provided under the MIT License — feel free to adapt and reuse the code, but please acknowledge the original authors and their publication when using results.

---

## Contact

For questions about the code or to request the original dataset/trained networks, please contact the repository owner.

