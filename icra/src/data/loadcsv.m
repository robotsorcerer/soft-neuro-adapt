cd('~/Desktop')

clear all;

exptIU = load('Exp1_cmd.csv');
exptIX = load('Exp1_states.csv');

N = exptIU(:,1);
exp_I.N = N;

exp_I.T1 = exptIU(:,2);
exp_I.cmd_vel_x1 = exptIU(:,3);
exp_I.cmd_vel_y1 = exptIU(:,4);
exp_I.pos_x1 = exptIX(:,3);
exp_I.pos_y1 = exptIX(:,4);
exp_I.est_vel_x1 = exptIX(:,5);
exp_I.est_vel_y1 = exptIX(:,6);

exp_I.T2 = exptIU(:,5);
exp_I.cmd_vel_x2 = exptIU(:,6);
exp_I.cmd_vel_y2 = exptIU(:,7);
exp_I.pos_x2 = exptIX(:,8);
exp_I.pos_y2 = exptIX(:,9);
exp_I.est_vel_x2 = exptIX(:,10);
exp_I.est_vel_y2 = exptIX(:,11);

exp_I.T3 = exptIU(:,8);
exp_I.cmd_vel_x3 = exptIU(:,9);
exp_I.cmd_vel_y3 = exptIU(:,10);
exp_I.pos_x3 = exptIX(:,13);
exp_I.pos_y3 = exptIX(:,14);
exp_I.est_vel_x3 = exptIX(:,15);
exp_I.est_vel_y3 = exptIX(:,16);

%%%%%%%%%%%%

exptIU = load('Exp2_cmd.csv');
exptIX = load('Exp2_states.csv');

N = exptIU(:,1);
exp_II.N = N;

exp_II.T1 = exptIU(:,2);
exp_II.cmd_vel_x1 = exptIU(:,3);
exp_II.cmd_vel_y1 = exptIU(:,4);
exp_II.pos_x1 = exptIX(:,3);
exp_II.pos_y1 = exptIX(:,4);
exp_II.est_vel_x1 = exptIX(:,5);
exp_II.est_vel_y1 = exptIX(:,6);

exp_II.T2 = exptIU(:,5);
exp_II.cmd_vel_x2 = exptIU(:,6);
exp_II.cmd_vel_y2 = exptIU(:,7);
exp_II.pos_x2 = exptIX(:,8);
exp_II.pos_y2 = exptIX(:,9);
exp_II.est_vel_x2 = exptIX(:,10);
exp_II.est_vel_y2 = exptIX(:,11);

exp_II.T3 = exptIU(:,8);
exp_II.cmd_vel_x3 = exptIU(:,9);
exp_II.cmd_vel_y3 = exptIU(:,10);
exp_II.pos_x3 = exptIX(:,13);
exp_II.pos_y3 = exptIX(:,14);
exp_II.est_vel_x3 = exptIX(:,15);
exp_II.est_vel_y3 = exptIX(:,16);

%%%%%%%%%%%%

exptIU = load('Exp3_cmd.csv');
exptIX = load('Exp3_states.csv');

N = exptIU(:,1);
exp_III.N = N;

exp_III.T1 = exptIU(:,2);
exp_III.cmd_vel_x1 = exptIU(:,3);
exp_III.cmd_vel_y1 = exptIU(:,4);
exp_III.pos_x1 = exptIX(:,3);
exp_III.pos_y1 = exptIX(:,4);
exp_III.est_vel_x1 = exptIX(:,5);
exp_III.est_vel_y1 = exptIX(:,6);

exp_III.T2 = exptIU(:,5);
exp_III.cmd_vel_x2 = exptIU(:,6);
exp_III.cmd_vel_y2 = exptIU(:,7);
exp_III.pos_x2 = exptIX(:,8);
exp_III.pos_y2 = exptIX(:,9);
exp_III.est_vel_x2 = exptIX(:,10);
exp_III.est_vel_y2 = exptIX(:,11);

exp_III.T3 = exptIU(:,8);
exp_III.cmd_vel_x3 = exptIU(:,9);
exp_III.cmd_vel_y3 = exptIU(:,10);
exp_III.pos_x3 = exptIX(:,13);
exp_III.pos_y3 = exptIX(:,14);
exp_III.est_vel_x3 = exptIX(:,15);
exp_III.est_vel_y3 = exptIX(:,16);

exp.expI = exp_I;
exp.expII = exp_II;
exp.expIII = exp_III;


%%%%%

save('exp.mat', 'exp')




