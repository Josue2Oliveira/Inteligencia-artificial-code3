
%PREDICT classica um conjunto de amostras utilizando os parâmetros da 
%rede (Theta1 e Theta2)
%   p = PREDICT(Theta1, Theta2, X) devolve um vetor com a classse de 
%   cada amostra do conjunto X.

% variaveis úteis


% Voce deve preencher o seguinte vetor corretamente com a classe de cada
% amostra
function p = predict(Theta1, Theta2, X)
a1 = [ones(size(X),1) X];
z2 = a1 * Theta1';
a2_ = [ones(size(z2),1) z2];      
a2 = sigmoid(a2_);          
z3 = a2 * Theta2';
a3 = sigmoid(z3);
[~,pos] = max(a3');
p = pos';
end