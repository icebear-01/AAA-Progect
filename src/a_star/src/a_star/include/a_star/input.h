#ifndef INPUT_H
#define INPUT_H

int input_step_positive(){
    static int k_step=0;
    if(k_step++<1000){
        return 0;
    }
    else if(k_step<1100){
        return 1.0*(k_step-1000);
    }
    else if(k_step<1600){
        return 100;
    } 
    else if(k_step<1700){
        return 100-1.0*(k_step-1600);
    }
    else return 0;
}

int input_step_negative(){
    static int k_step=0;
    if(k_step++<1000){
        return 0;
    }
    else if(k_step<1100){
        return -1.0*(k_step-1000);
    }
    else if(k_step<1600){
        return -100;
    } 
    else if(k_step<1700){
        return -100+1.0*(k_step-1600);
    }
    else return 0;
}

int input_pulse_positive(){
    static int k_pulse=0;
    static int cnt=0;
    if(k_pulse>7){
        k_pulse=0;
        cnt++;
    }
    if(cnt<4){
        if(k_pulse++<3){
            return 0;
        }
        else if(k_pulse<5){
            return 25*(k_pulse-3);
        }
        else if(k_pulse<7){
            return 50-25*(k_pulse-5);
        }
        else return 0;
    }
    else return 0;
}

int input_pulse_negative(){
    static int k_pulse=0;
    static int cnt=0;
    if(k_pulse>7){
        k_pulse=0;
        cnt++;
    }
    if(cnt<4){
        if(k_pulse++<3){
            return 0;
        }
        else if(k_pulse<5){
            return -25*(k_pulse-3);
        }
        else if(k_pulse<7){
            return -50+25*(k_pulse-5);
        }
        else return 0;
    }
    else return 0;
}

int input_sin(){
    static double k_sin=-0.2;
    k_sin+=0.2;
    double wt=3.1415926*k_sin/50.0;
    return 80*sin(wt);
}

#endif // INPUT_H
