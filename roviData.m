clear;
clc;
close all;

files = dir('qFile_3markers_Fast_*.csv');
Q3F = cell(length(files),1);
for k = 1:length(files) 
  Q3F{k} = csvread(files(k).name);
end
files = dir('qFile_3markers_Medi_*.csv');
Q3M = cell(length(files),1);
for k = 1:length(files) 
  Q3M{k} = csvread(files(k).name);
end
files = dir('qFile_3markers_Slow_*.csv');
Q3S = cell(length(files),1);
for k = 1:length(files) 
  Q3S{k} = csvread(files(k).name);
end
files = dir('qFile_1marker_Fast_*.csv');
Q1F = cell(length(files),1);
for k = 1:length(files) 
  Q1F{k} = csvread(files(k).name);
end
files = dir('qFile_1marker_Medi_*.csv');
Q1M = cell(length(files),1);
for k = 1:length(files) 
  Q1M{k} = csvread(files(k).name);
end
files = dir('qFile_1marker_Slow_*.csv');
Q1S = cell(length(files),1);
for k = 1:length(files) 
  Q1S{k} = csvread(files(k).name);
end
files = dir('qFile_vis_Fast_*.csv');
QVF = cell(length(files),1);
for k = 1:length(files) 
  QVF{k} = csvread(files(k).name);
end
files = dir('qFile_vis_Medi_*.csv');
QVM = cell(length(files),1);
for k = 1:length(files) 
  QVM{k} = csvread(files(k).name);
end
files = dir('qFile_vis_Slow_*.csv');
QVS = cell(length(files),1);
for k = 1:length(files) 
  QVS{k} = csvread(files(k).name);
end

files = dir('toolPosFile_3markers_Fast_*.csv');
T3F = cell(length(files),1);
for k = 1:length(files) 
  T3F{k} = csvread(files(k).name);
end
files = dir('toolPosFile_3markers_Medi_*.csv');
T3M = cell(length(files),1);
for k = 1:length(files) 
  T3M{k} = csvread(files(k).name);
end
files = dir('toolPosFile_3markers_Slow_*.csv');
T3S = cell(length(files),1);
for k = 1:length(files) 
  T3S{k} = csvread(files(k).name);
end
files = dir('toolPosFile_1marker_Fast_*.csv');
T1F = cell(length(files),1);
for k = 1:length(files) 
  T1F{k} = csvread(files(k).name);
end
files = dir('toolPosFile_1marker_Medi_*.csv');
T1M = cell(length(files),1);
for k = 1:length(files) 
  T1M{k} = csvread(files(k).name);
end
files = dir('toolPosFile_1marker_Slow_*.csv');
T1S = cell(length(files),1);
for k = 1:length(files) 
  T1S{k} = csvread(files(k).name);
end
files = dir('toolPosFile_vis_Fast_*.csv');
TVF = cell(length(files),1);
for k = 1:length(files) 
  TVF{k} = csvread(files(k).name);
end
files = dir('toolPosFile_vis_Medi_*.csv');
TVM = cell(length(files),1);
for k = 1:length(files) 
  TVM{k} = csvread(files(k).name);
end
files = dir('toolPosFile_vis_Slow_*.csv');
TVS = cell(length(files),1);
for k = 1:length(files) 
  TVS{k} = csvread(files(k).name);
end

files = dir('trackErrorFile_3markers_Fast_*.csv');
E3F = cell(length(files),1);
for k = 1:length(files) 
  E3F{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_3markers_Medi_*.csv');
E3M = cell(length(files),1);
for k = 1:length(files) 
  E3M{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_3markers_Slow_*.csv');
E3S = cell(length(files),1);
for k = 1:length(files) 
  E3S{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_1marker_Fast_*.csv');
E1F = cell(length(files),1);
for k = 1:length(files) 
  E1F{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_1marker_Medi_*.csv');
E1M = cell(length(files),1);
for k = 1:length(files) 
  E1M{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_1marker_Slow_*.csv');
E1S = cell(length(files),1);
for k = 1:length(files) 
  E1S{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_vis_Fast_*.csv');
EVF = cell(length(files),1);
for k = 1:length(files) 
  EVF{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_vis_Medi_*.csv');
EVM = cell(length(files),1);
for k = 1:length(files) 
  EVM{k} = max(csvread(files(k).name));
end
files = dir('trackErrorFile_vis_Slow_*.csv');
EVS = cell(length(files),1);
for k = 1:length(files) 
  EVS{k} = max(csvread(files(k).name));
end

%%
C = T1F(1,1,1,:)
M = cell2mat(C)
M(1,1)

