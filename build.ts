import { GImage, GImageSet, FIELDS } from "./types.js"

function loadImageSets(filenames: string[]): Map<string, GImageSet> {
    const RE_FILENAME = /(.+[\d_]+)--(.+)--([\w\+\d_,]+)\/\d+\.(\d+)\.png/
    const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/
    const RE_MODEL = /([\w\d\._-]+)_r(\d+)_(\d+)/
    
    var imageSets = new Map<string, GImageSet>()
    for (const filename of filenames) {
        var match = RE_FILENAME.exec(filename)
        if (match) {
            var modelStr = match[1]
            var prompt = match[2]
            var samplerStr = match[3]
            var seed = parseInt(match[4])

            var sampler = ""
            var samplerSteps = 0
            var cfg = 0
            match = RE_SAMPLER.exec(samplerStr)
            if (match) {
                sampler = match[1]
                samplerSteps = parseInt(match[2])
                cfg = parseInt(match[3])
            }

            var modelName = modelStr
            var modelSeed = 0
            var modelSteps = 0
            match = RE_MODEL.exec(modelStr)
            if (match) {
                modelName = match[1]
                modelSeed = parseInt(match[2])
                modelSteps = parseInt(match[3])
            }

            var iset = new GImageSet(modelName, modelSeed, modelSteps, prompt, sampler, samplerSteps, cfg)
            var isetKey = iset.getKey(FIELDS)
            if (imageSets.has(isetKey)) {
                iset = imageSets.get(isetKey) as GImageSet
            }
            else {
                imageSets.set(isetKey, iset)
            }

            // add an image to the imageset.
            iset.images.push(new GImage(filename, seed))
        }
        else {
            console.log(`no match: ${filename}`)
        }
    }

    for (const iset of imageSets.values()) {
        iset.images = iset.images.sort((a, b) => a.filename.localeCompare(b.filename))
    }
    return imageSets
}

export { loadImageSets }
