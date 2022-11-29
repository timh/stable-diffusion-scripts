class GImage {
    filename: string
    seed: number
}

var sortKeys = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
class GImageSet {
    modelName: string
    modelSeed: number = 0
    modelSteps: number = 0
    prompt: string
    sampler: string
    samplerSteps: number = 0
    cfg: number = 0

    images: Array<GImage> = []

    getKey(): string {
        var res = ""
        sortKeys.forEach((key) => {
            if (res) {
                res += ", "
            }
            var val = this[key]
            if (typeof val == "number") {
                val = val.toString().padStart(5, "0")
            }
            else {
                val = val.toString()
            }
            res += (key + "=" + val)
        })
        return res
    }
}

var imageSets = new Map<string, GImageSet>()

const RE_FILENAME = /(.+[\d_]+)--(.+)--([\w\+\d_,]+)\/\d+\.(\d+)\.png/
const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/
const RE_MODEL = /([\w\d\._-]+)_r(\d+)_(\d+)/

async function updateList() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")
        filenames.forEach((filename) => {
            var match = RE_FILENAME.exec(filename)
            if (match) {
                var iset = new GImageSet()
                var img = new GImage()
                img.filename = filename

                var modelStr = match[1]
                iset.prompt = match[2]
                var samplerStr = match[3]
                var seedStr = match[4]
                img.seed = parseInt(seedStr)

                match = RE_SAMPLER.exec(samplerStr)
                if (match) {
                    iset.sampler = match[1]
                    iset.samplerSteps = parseInt(match[2])
                    iset.cfg = parseInt(match[3])
                }

                var modelSeed = 0
                var modelSteps = 0
                match = RE_MODEL.exec(modelStr)
                if (match) {
                    iset.modelName = match[1]
                    iset.modelSeed = parseInt(match[2])
                    iset.modelSteps = parseInt(match[3])
                }
                else {
                    iset.modelName = modelStr
                }

                var isetKey = iset.getKey()
                if (imageSets.has(isetKey)) {
                    iset = imageSets.get(isetKey) as GImageSet
                }
                else {
                    imageSets.set(isetKey, iset)
                }

                // add an image to the imageset.
                iset.images.push(img)
            }
        })

        var sortedKeys = Array.from(imageSets.keys()).sort()
        // imageSets.forEach((val, key) => {
        //     console.log(`${key} has ${val.images.length}`)
        // });
        sortedKeys.forEach((key) => {
            var val = imageSets.get(key)
            console.log(`${key} has ${val!.images.length}`)
    })
        //console.log(imageSets.keys())
    }
    else {
        console.log(`error`)
    }
    return null;
}

updateList().then((val) => {
    console.log("done")
})
