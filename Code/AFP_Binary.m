function y = AFP_Binary(id_AFP,XX_ML,XX_CV,CV)
%     id_AFP = indxFinder('AFP');
if CV
    
    a = XX_ML{:,id_AFP};
    b = mean(a);

    c = XX_CV{:,id_AFP};
    d = mean(c);

    id_con_100 = a-d > 100-d;
    id_con_8_100 = a-d > 8-d & a-d <= 100-d;
    id_con_8 = a-b <= 8-d;
else
    id_con_100 = XX_ML{:,id_AFP} > 100;
    id_con_8_100 = XX_ML{:,id_AFP} > 8 & XX_ML{:,id_AFP} <= 100;
    id_con_8 = XX_ML{:,id_AFP} <= 8;
end

XX_ML{id_con_100,id_AFP} = 2;
XX_ML{id_con_8_100,id_AFP} = 1;
XX_ML{id_con_8,id_AFP} = 0;

y = XX_ML;
end