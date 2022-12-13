import { GImage, GImageSet, FIELDS } from "./types.js"

function loadImageSets(filenames: string[]): Map<string, GImageSet> {
    const RE_FILENAME = /([^\/]+)--(.+)--(.+)\/\d+\.(\d+)\.png/
    const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/

    // alex44-0.9e-6-f222_r0_9000
    const RE_MODEL = /^([\w\d_\.\-\+]+)_r(\d+)_(\d+)$/

    // alex44-everydream-e01_00440
    // output_alex22_768-sd21@4.0_3300
    const RE_MODEL_EVERYDREAM = /^([\w\d@_\-\+\.]+)_(\d+)$/

    const RE_MODEL_BATCH = /^(.+)-batch(\d+)(.*)$/
    const RE_MODEL_LR = /^(.+)@([\d\.]+)(.*)$/
    
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
            var modelBatch = 1
            var modelLR = "1.0"
            match = RE_MODEL.exec(modelStr)
            if (match) {
                modelName = match[1]
                modelSeed = parseInt(match[2])
                modelSteps = parseInt(match[3])
            }
            else {
                match = RE_MODEL_EVERYDREAM.exec(modelStr)
                if (match) {
                    modelName = match[1]
                    modelSteps = parseInt(match[2])
                }
            }

            const modelNameOrig = modelName
            match = RE_MODEL_BATCH.exec(modelName)
            if (match) {
                modelBatch = parseInt(match[2])
                modelName = match[1] + match[3]
            }

            match = RE_MODEL_LR.exec(modelName)
            if (match) {
                modelLR = match[2]
                modelName = match[1] + match[3]
            }

            modelName = modelName.replace("-batch", " batch")
            modelName = modelName.replace("-cap", " +cap")
            modelName = modelName.replace("-bf16", "")
            modelName = modelName.replace("-f222", " f222")
            modelName = modelName.replace("-f222v", " f222v")
            modelName = modelName.replace("-warmup", " warmup")
            modelName = modelName.replace("_r0", "") // temp

            var iset = new GImageSet({modelName: modelName, modelNameOrig: modelNameOrig, modelSeed: modelSeed, modelSteps: modelSteps, 
                                      modelBatch: modelBatch, modelLR: modelLR,
                                      prompt: prompt, 
                                      sampler: sampler, samplerSteps: samplerSteps, cfg: cfg})
            // var iset = new GImageSet(modelName, modelSeed, modelSteps, prompt, sampler, samplerSteps, cfg)
            const isetKey = iset.getKey(FIELDS)
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
