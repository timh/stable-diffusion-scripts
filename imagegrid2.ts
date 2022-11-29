interface ImageSet {
    model_name: string
    model_seed: string
    model_steps: number
    prompt: string
    sampler: string
    sampler_steps: number
    cfg: number

    key: string
}

interface Image {
    picset: ImageSet
    filename: string
    seed: number
}

var models = Array<Model>()
var imageSets = Array<ImageSet>()
var images = Array<Image>()

const RE_FILENAME = /(.+[\d_]+)--(.+)--([\w\+\d_,]+)\/\d+\.(\d+)\.png/
const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/
const RE_MODEL = /([\w\d\._-]+)_r(\d+)_(\d+)/

async function updateList() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var lines = data.split("\n")
        lines.forEach((line) => {
            var match = RE_FILENAME.exec(line);
            if (match) {
                var model_str = match[1]
                var prompt = match[2]
                var samplerStr = match[3]
                var seedStr = match[4]
                var seed = parseInt(seedStr)
                var sampler = samplerStr
                var samplerSteps = 0
                var cfg = 0

                match = RE_SAMPLER.exec(samplerStr)
                if (match) {
                    sampler = match[1]
                    samplerSteps = parseInt(match[2])
                    cfg = parseInt(match[3])
                }
                console.log(`model_str ${model_str}, prompt '${prompt}', sampler ${sampler} ${samplerSteps}, cfg ${cfg}, seed ${seed}`)
            }
        })
    }
    else {
        console.log(`error`)
    }
    return null;
}

updateList().then((val) => {
    console.log("done")
})
