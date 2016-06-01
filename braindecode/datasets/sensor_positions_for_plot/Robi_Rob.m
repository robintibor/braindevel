% 6.1) lädt die 2d projezierten Elektrodenkoordinaten der EEG Kappe und
load('H:\Laura\SEPStudie\waveguard2Dpos.mat');
x=num(:,1);
y=num(:,2);

Kanalnamenchar = txt;
%% 6.2) transformiert diese in die richtige Richtung (Nase nach oben)
%Ergebnis der Rotationsmatrix TM=[cos(phi), -sin(phi);sin(phi), cos(phi)]
%für phi=90° (TMplus90) bzw phi=-90 (TMminus90)

TMplus90=[0,-1;1,0];
xy=[x,y];
for i=1:size(xy,1);
    xytrafo(i,1:2) = TMplus90*(xy(i,:)');%#ok
end

% 6.3) da Cz nicht genau auf (0,0) liegt sondern bei (0.1, -16.8) werden alle
%Koordinaten um diesen Wert verschoben, so dass Cz genau im Ursprung liegt
xx=xytrafo(:,1);
xv= xx-0.1;
yy=xytrafo(:,2);
yv=yy+16.8;

xy=[xv,yv];

%6.4) Normierung aller Elektrodenkoordinaten auf 1
%6.4.1) alle positiven y -Werte auf y max normieren und in xy den normierten Wert reinschreiben
indypos=find(xy(:,2)>=0);
ypos=xy(indypos,2);
ynormpos=ypos/(max(xy(:,2)));
YNPI=[];
if indypos~=0;
    for j=1:size(indypos,1);
        ynpi=ynormpos(j);
        YNPI=[YNPI;ynpi];%#ok
        %%%%%j=j+1;
    end
    xy(indypos,2)=YNPI;
end

%6.4.2) alle negativen y -Werte auf y min normieren und in xy den normierten Wert reinschreiben
indyneg=find(xy(:,2)<0);
yneg=xy(indyneg,2);
ynormneg=yneg/(min(xy(:,2)));
YNNI=[];
if indyneg~=0;
    for j=1:size(indyneg,1);
        ynni=-ynormneg(j);
        YNNI=[YNNI;ynni];%#ok
    end
    xy(indyneg,2)=YNNI;
end

%6.4.3) alle positiven x -Werte auf x max normieren und in xy den normierten Wert reinschreiben
indxpos=find(xy(:,1)>=0);
xpos=xy(indxpos,1);
xnormpos=xpos/(max(xy(:,1)));
XNPI=[];
if indxpos~=0;
    for j=1:size(indxpos,1);
        xnpi=xnormpos(j);
        XNPI=[XNPI;xnpi];%#ok
    end
    xy(indxpos,1)=XNPI;
end

%6.4.4)alle negativen x -Werte auf x min normieren und in xy den normierten Wert reinschreiben
indxneg=find(xy(:,1)<0);
xneg=xy(indxneg,1);
xnormneg=xneg/(min(xy(:,1)));
XNNI=[];
if indxneg~=0;
    for j=1:size(indxneg,1);
        xnni=-xnormneg(j);
        XNNI=[XNNI;xnni];%#ok
    end
    xy(indxneg,1)=XNNI;
end

%8)tatsächliche Elektroden, die außerhalb des Einheitskreises liegen auf Einheitskreis
%projezieren%das sind die die du umbauen mußt:

FT10=strmatch('FT10',Kanalnamenchar);%#ok
%FT10hisKoord=xytrafo(66,:)
xy(66,:)=[0.887,0.4618];

FT9=strmatch('FT9',Kanalnamenchar);%#ok %gibt an welcher Stelle FT9 steht
% FT9isKoord=xytrafo(65,:) %ursprüngliche Koordianten von FT9
xy(65,:)=[-0.887,0.4618];
%neue Koordinaten von FT9 (manuell gewählt)

M2=strmatch('M2',Kanalnamenchar);%#ok
% M1hisKoord=xytrafo(19,:)
xy(19,:)=[0.9573,-0.289];

M1=strmatch('M1',Kanalnamenchar);%#ok
% M1hisKoord=xytrafo(13,:)
xy(13,:)=[-0.9573,-0.289];

xv=xy(:,1); % x-Koordinaten der Elektroden
yv=xy(:,2); % y-Koordinaten der Elektroden

fileId = fopen('sensor-pos-laura.txt', 'w');
for iSensor = 1:size(xy,1)
    fprintf(fileId, '(''%s'', (%.3f, %.3f)),\n', Kanalnamenchar{iSensor}, ...
        xy(iSensor,1), xy(iSensor,2));
end
fclose(fileId);
