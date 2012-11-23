

figure; set(gcf, 'color', 'w');

% hamster
at = {'golden hamster'
'hamster'
'rodent'
'mammal'
'fauna, animal'};

a = [1	0.0133
0	0.2392
-1	0.1134
-2	0.011
-6	0.072];
subplot(411); 
bar(a(:,1), a(:, 2)); grid on; axis([-7 2 -inf max(c(:,2))+.02])
text(a(:,1), a(:,2)+.01, at, 'Rotation', 30);



% hydroplane 
bt = {'hydroplane,hydrofoil'
'speedboat'
'powerboat'
'boat'};

b = [0	0.2857
-1	0.0105
-2	0.0139
-3	0.1289];
subplot(412); 
bar(b(:,1), b(:, 2)); grid on; axis([-7 2 -inf max(b(:,2))+.02])
text(b(:,1), b(:,2)+.01, bt, 'Rotation', 30);

% woolly bear caterpillar
ct = {'wooly bear moth'
'woolly bear caterpillar'
'caterpillar'
'larva'
'fauna, animal'};

c = [1	0.0697
0	0.0657
-1	0.1833
-2	0.0147
-3	0.0198];
subplot(413); 
bar(c(:,1), c(:, 2)); grid on; axis([-7 2 -inf max(c(:,2))+.02])
text(c(:,1), c(:,2)+.01, ct, 'Rotation', 30);

% cottage cheese
dt = {'cottage cheese'
'cheese'
'diary food'
'food' };

d = [0	0.0449
-1	0.0654
-2	0.0098
-3	0.1435];
subplot(414); 
bar(d(:,1), d(:, 2)); grid on; axis([-7 2 -inf max(d(:,2))+.02 ])
text(d(:,1), d(:,2)+.01, dt, 'Rotation', 30);

saveas(gcf, '/Users/xlx/proj/ImageNet/category_base.png')


% -------------- chairs -------
figure; set(gcf, 'color', 'w');

% folding chair
et = {'beach/camp chair'
'folding chair'
'chair'
'seat'
'furniture' };

e =[1	0.0208
0	0.1726
-1	0.0715
-2	0.007
-3	0.0129];

subplot(211); 
bar(e(:,1), e(:, 2)); grid on; axis([-4 2 0 max(e(:,2))+.02 ])
text(e(:,1), e(:,2)+.01, et, 'Rotation', 30);

ft = {'side chair'
'chair'
'seat'
'furniture'
};
f = [0	0.0696
-1	0.2154
-2	0.00138
-3	0.0552];

subplot(212); 
bar(f(:,1), f(:, 2)); grid on; axis([-4 2 0 max(f(:,2))+.02 ])
text(f(:,1), f(:,2)+.01, ft, 'Rotation', 30);
