//#include <stdio.h>
/* we include the above library for basic functionalities relatedwith i/o */
#include <stdio.h>

/*
int main(void){
	printf("Hello, it all started here \n");
}*/

/*
int main (int argc, char *argv[]){
	printf("Hello, it all started here \n");
}*/

/*
int main(void){
	printf("Hello,");
	printf("it all started here");
	printf("\n");
}*/

int main(void){
	int count = 0;
	char *statement = "Hello, it all started here\n";

	while(statement[count] != '\0') //is non-zero (NUL character) 
		printf("%c", statement[count++]); 
	return 0;
}
//Non-zero corresponds to TRUE and zero to FALSE

