library(stringr)
library(magrittr)
df0 = read.csv('./data.csv', header = F)
d = c()
for(y in 1:ncol(df0)){
  # dy = data.frame()
  for(x in 1:nrow(df0)){
    a = str_split(df0[x, y], '\n')[[1]] %>% str_c(collapse = '') %>% str_remove_all('\\[|\\]')
    b = str_split(a, ' ')[[1]]
    c = b[!(b %in% c('[', ']', ''))]
    # dy = rbind(dy, c)
    d = c(d, c)
  }
}

m = array(d, dim=c(70,63,40))
m[, , 1] %>% dim()
tran_m=aperm(m,c(3,1,2))
dim(tran_m)
cut_m=tran_m[,1:3,1:3]
c=as.numeric(cut_m)
a=array(c,dim=c(40,3,3))
