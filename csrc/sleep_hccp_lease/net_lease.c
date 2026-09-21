/* Investigation shim: process-local HCCP lifecycle lease, no communicator.
 * Not a public CANN API. Load only into the isolated test worker. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <driver/ascend_hal.h>

typedef struct { const char *paramInfo; uint64_t paramLen; } ExtParam;
typedef struct { ExtParam *extParamList; uint64_t extParamCnt; } OpenArgs;
typedef int (*OpenFn)(const OpenArgs *);
typedef int (*CloseFn)(void);
typedef struct { int active, held, deferred, opens, closes, reused; char args[256]; } State;
static State states[64];
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static int device(void) {
    int dev = -1;
    int (*get)(int *) = dlsym(RTLD_DEFAULT, "rtGetDevice");
    return get && get(&dev) == 0 && dev >= 0 && dev < 64 ? dev : -1;
}
static void trace(const char *event, int d, int rc) {
    State *s=&states[d];
    fprintf(stderr,"NET_LEASE pid=%d dev=%d event=%s rc=%d active=%d held=%d deferred=%d opens=%d closes=%d reused=%d\n",
        getpid(),d,event,rc,s->active,s->held,s->deferred,s->opens,s->closes,s->reused);
    fflush(stderr);
}
int rtOpenNetService(const OpenArgs *args) {
    OpenFn real=(OpenFn)dlsym(RTLD_NEXT,"rtOpenNetService");
    int d=device(); if (!real || d<0) return 107000;
    pthread_mutex_lock(&mutex); State *s=&states[d];
    const char *param=args && args->extParamCnt==1 ? args->extParamList[0].paramInfo : NULL;
    uint64_t n=param ? args->extParamList[0].paramLen : 0;
    int rc;
    if (s->held && s->deferred) {
        /* Fail closed if the requested service configuration changed. */
        if (!param || n>=sizeof(s->args) || strlen(s->args)!=n || memcmp(s->args,param,n)) rc=107000;
        else {s->deferred=0; s->reused++; rc=0;}
        trace("reuse_open",d,rc);
    } else {
        rc=real(args);
        if (!rc) {s->active=1;s->opens++; if(param && n<sizeof(s->args)){memcpy(s->args,param,n);s->args[n]=0;}}
        trace("real_open",d,rc);
    }
    pthread_mutex_unlock(&mutex);return rc;
}
int rtCloseNetService(void) {
    CloseFn real=(CloseFn)dlsym(RTLD_NEXT,"rtCloseNetService");
    int d=device();if(!real || d<0)return 107000;
    pthread_mutex_lock(&mutex);State *s=&states[d];int rc;
    if(s->held && s->active && !s->deferred){s->deferred=1;rc=0;trace("defer_close",d,rc);}
    else if(s->held){rc=107000;trace("invalid_close",d,rc);}
    else {rc=real();if(!rc){s->active=0;s->closes++;}trace("real_close",d,rc);}
    pthread_mutex_unlock(&mutex);return rc;
}
int noanchor_hold(void) {
    int d=device();if(d<0)return -1;
    pthread_mutex_lock(&mutex);State *s=&states[d];
    int rc=(!s->active || s->held || s->deferred) ? -2:0;
    if(!rc)s->held=1;trace("hold",d,rc);pthread_mutex_unlock(&mutex);return rc;
}
int noanchor_release(void) {
    int d=device();if(d<0)return -1;
    pthread_mutex_lock(&mutex);State *s=&states[d];int rc=0;
    if(!s->held)rc=-2;
    else if(s->deferred){CloseFn real=(CloseFn)dlsym(RTLD_NEXT,"rtCloseNetService");rc=real?real():-3;
        if(!rc){s->active=0;s->deferred=0;s->held=0;s->closes++;}}
    else s->held=0;
    trace("release",d,rc);pthread_mutex_unlock(&mutex);return rc;
}
int noanchor_query_pid(int dev, int type, int *pid) {
    void *lib=dlopen("libascend_hal.so",RTLD_NOW);
    if(!lib)return -1;
    drvError_t (*query)(struct halQueryDevpidInfo,pid_t *)=dlsym(lib,"halQueryDevpid");
    struct halQueryDevpidInfo q={0};q.hostpid=getpid();q.devid=dev;q.proc_type=type;
    int rc=query?query(q,pid):-2;dlclose(lib);return rc;
}
