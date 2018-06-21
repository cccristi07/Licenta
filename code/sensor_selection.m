clear; clc;

%% înc?rcare set de date hanoi
load net_simulations

%% extragem referin?ele
%% extragere referinta
ref_pressure = train_data.NODE_VALUES{1}.EN_PRESSURE; % nu exista nici o scurgere
ref_demand = train_data.NODE_VALUES{1}.EN_DEMAND; %
ref_velocity = train_data.LINK_VALUES{1}.EN_VELOCITY;

node_values = train_data.NODE_VALUES;
link_values = train_data.LINK_VALUES;

ref_pressure_mean = mean(ref_pressure(1:35,:),1);
ref_demand_mean = mean(ref_demand(1:35, :),1);
ref_velocity_mean = mean(ref_velocity(1:35, :),1);

leg = {};

for i = 1:31
    leg{i} = sprintf('NODE %d', i);
end


%% calculam matricea de reziduuri scalate in [0, 1] astfel încât diferentele cele mai mari fata de referinta sa corespunda
% valorilor apropiate de 1 

% aleg un emitter standard de 25 pentru a crea matricea de Reziduuri R care
% are dimensiunile (n_faults, n_nodes) n_faults reprezinta emitter_ul
% simulat în fiecare nod al retelei
emitter_val = 25;
node_vals = train_data.NODE_VALUES;
R = [];
for i = 1:31
    
    sim_vals = get_emitter_vals(node_vals, emitter_val, i);
    measured_pressure = sim_vals.EN_PRESSURE;
    scores = abs_residual(ref_pressure, measured_pressure);
    rez = normc(scores);
    rez = mean(scores(2:25, :));
    rez = abs(rez);
    rez = (rez - min(rez))/(max(rez)-min(rez));
    R = [R rez'];
end

imshow(R')
R = R'; % consideram linia_i ca fiind rasp tuturor nodurilor la fault_i 

%% constructie matrice binara M
% fault detection via set covering problem
% care sunt cele mai importante noduri pentru detectia fault-ului


thr = 0.65; % setarea limitei de sensibilitate a senzorilor
M = double(R > thr); % binarizarea senzorilor
alpha = binvar(size(M,2), 1); % definirea variabilei binare alpha 

constrangere = [M * alpha >=1]; 
obiectiv = sum(alpha); % criteriul de minimizare

optimize(constrangere, obiectiv)
alpha = value(alpha);
sum(alpha); % numarul de noduri care algoritmul de optimizare numerica a selectat
sum(M(alpha == 1, :),2) >= 1; % verif daca se indeplineste conditia de set covering
find(alpha) % nodurile selectate de algoritm