%%Author: Philip Giles, UCD
function [output_images,output_labels] = selectRelevantData(input_images,input_labels, Num1, Num2)
%Function returns images and labels only of the specified digit 

numData = size(input_labels)

j = 1;
for i = 1:1:numData
    if((input_labels(i) == Num1)||(input_labels(i) == Num2))
        output_images(:,:,j) = input_images(:,:,i);
        output_labels(j,:) = input_labels(i,:);
        j = j+1;
    end
end

end

