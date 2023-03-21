import argparse
import logging
import torch
from diffusers import StableDiffusionPipeline
import os

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Model path")
    parser.add_argument("--input_text", type=str, default="a drawing of a gray and black dragon", help="input text")
    parser.add_argument("--output_dir", type=str, default="saved_results",help="output path")
    parser.add_argument("--base_images", type=str, default="base_images/image.jpg", help="Path to training images for FID input")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of images to generate per prompt")
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf16, fp16')
    parser.add_argument('--ipex', action='store_true', default=False, help='ipex')
    parser.add_argument('--trace', action='store_true', default=False, help='jit trace')
    parser.add_argument('--compile_ipex', action='store_true', default=False, help='compile with ipex backend')
    parser.add_argument('--compile_inductor', action='store_true', default=False, help='compile with inductor backend')
    parser.add_argument('--profile', action='store_true', default=False, help='profile')
    parser.add_argument('--benchmark', action='store_true', default=False, help='test performance')
    parser.add_argument('--accuracy', action='store_true', default=False, help='test accuracy')
    
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    logger.info(f"Parameters {args}")
    
    # load model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name_or_path)
    
    # print model
    print("pipe.unet", pipe.unet)

    # data type
    if args.precision == "fp32":
        dtype=torch.float32
    elif args.precision == "bf16":
        dtype=torch.bfloat16
    elif args.precision == "fp16":
        dtype=torch.half
    else:
        raise ValueError("--precision needs to be the following:: fp32, bf16, fp16")
    # ipex
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        if args.precision == "fp32":
            pipe.unet = ipex.optimize(pipe.unet.eval(), inplace=True)
        elif args.precision == "bf16" or args.precision == "fp16":
            pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=dtype, inplace=True)
        else:
            raise ValueError("--precision needs to be the following:: fp32, bf16, fp16")
        print("Running IPEX ...")
    
    
    # jit trace
    if args.trace:
        # from utils_vis import make_dot, draw
        if args.precision == "bf16" or args.precision == "fp16":
            with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.traced_unet, (torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768)), strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                pipe.traced_unet(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                graph = pipe.traced_unet.graph_for(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                print(graph)
                # draw(graph).render("stable_diffusion")

        else:
            with torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.traced_unet, (torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768)), strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                pipe.traced_unet(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                graph = pipe.traced_unet.graph_for(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                print(graph)
                # draw(graph).render("stable_diffusion")
    
    # torch compile with ipex backend
    if args.compile_ipex:
        pipe.unet = torch.compile(pipe.unet, backend='ipex')
    # torch compile with inductor backend
    if args.compile_inductor:
        pipe.unet = torch.compile(pipe.unet, backend='inductor')
    
    # create input
    prompts = []
    for _ in range(args.batch_size):
        prompts.append(args.input_text)
    print("prompts", prompts)
    
    # warm up
    if args.precision == "bf16" or args.precision == "fp16":
        with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
            images = pipe(prompts).images
    else:
        with torch.no_grad():
            images = pipe(prompts).images
    
    # run model
    if args.precision == "bf16" or args.precision == "fp16":
        with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
            images = pipe(prompts).images
    else:
        with torch.no_grad():
            images = pipe(prompts).images


    # profile
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
            if args.precision == "bf16" or args.precision == "fp16":
                with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                    images = pipe(prompts).images
            else:
                with torch.no_grad():
                    images = pipe(prompts).images
            
        output = p.key_averages().table(sort_by="self_cpu_time_total")
        print(output)
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            try:
                os.makedirs(timeline_dir)
            except:
                pass
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    'stable_diffusion-' + '-' + str(os.getpid()) + '.json'
        p.export_chrome_trace(timeline_file)

if __name__ == '__main__':
    main()