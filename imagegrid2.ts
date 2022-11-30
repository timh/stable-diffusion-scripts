import { GImage, GImageSet, ColumnHeader, sort } from "./types"
import { buildImageSets } from "./build"

// var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'samplerStr', 'cfg']
var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'samplerStr', 'cfg']

var allImageSets: Map<string, GImageSet>              // image sets by key
var allImageSetKeys: Array<string>                    // sorted
var uniqueFieldValues: Map<string, Array<Object>>     // unique sorted values for each field
var fieldValueIndex: Map<String, Map<Object, number>> // index in uniqueFieldValues for each field, value



function buildHeaders(imageSetKeys: string[]): ColumnHeader[] {
    var lastHeaders = new Map<string, ColumnHeader>() // current header by field
    var allHeaders = new Array<ColumnHeader>()

    // walk through image sets in order, building the columns out
    imageSetKeys.forEach((setKey) => {
        var imgSet = allImageSets.get(setKey) as GImageSet

        for (const [idx, field] of fields.entries()) {
            var header = lastHeaders.get(field)
            if (header == null || header?.value != imgSet![field]) {
                var column = (header != null) ? header.columnEnd : 2
                header = new ColumnHeader(idx + 1, field, imgSet![field], column)
                lastHeaders.set(field, header)
                allHeaders.push(header)
            }
            header.columnEnd ++
        }
    })
    return allHeaders
}

function toggle(span: Element, field: string, idx: number) {
    var className = `${field}_${idx}`
    var index = span.className.indexOf('selected') 
    var hide = (index != undefined && index != -1)
    span.className = hide ? "" : "selected"

    for (const el of document.getElementsByClassName(className)) {
        if (hide) {
            el.className = el.className + " hidden"
        }
        else {
            el.className = el.className.replace(" hidden", "")
        }
    }
}

function renderChoices(field: string) {
    var allUnique = new Set()
    var isNumber = false
    for (const obj of allImageSets.values()) {
        allUnique.add(obj[field])
        if (typeof obj[field] == 'number') {
            isNumber = true
        }
    }

    var choices = sort(allUnique)

    var html = ""
    html += `<span class="field">${field}</span>\n`
    html += `<span class="values">\n`
    for (const [idx, choice] of choices.entries()) {
        html += `  <span class="selected" onClick="toggle(this, '${field}', ${idx})">${choice}</span>\n`
    }
    html += `</span><!-- values -->\n`

    document.getElementById('chooser')!.innerHTML += html
}

async function updateImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        allImageSets = buildImageSets(fields, filenames)
        allImageSetKeys = sort(allImageSets.keys()) as string[]

        // build sorted list of unique values for each field.
        // start by building a set.
        var uniqueFieldValuesSet = new Map<string, Set<Object>>()
        for (const field of fields) {
            uniqueFieldValuesSet.set(field, new Set<Object>())
            for (const iset of allImageSets) {
                uniqueFieldValuesSet.get(field)!.add(iset[field])
            }
        }

        // then convert it to a sorted array
        uniqueFieldValues = new Map<string, Array<Object>>()
        for (const field of fields) {
            var val = uniqueFieldValuesSet.get(field)!
            uniqueFieldValues.set(field, sort(val))
        }

        fieldValueIndex = new Map<string, Map<Object, number>>()
        for (const field of fields) {
            fieldValueIndex.set(field, new Map<string, number>())
            for (const [idx, value] of uniqueFieldValues.get(field)!.entries()) {
                fieldValueIndex.get(field)!.set(value, idx)
            }
        }
    }
    else {
        console.log("error")
    }
}

async function update() {
    await updateImages()

    // renderChoices('modelStr')
    renderChoices('modelName')
    renderChoices('modelSeed')
    renderChoices('modelSteps')
    renderChoices('prompt')

    var allHeaders = buildHeaders(allImageSetKeys)
    var grid = document.getElementById("imagegrid") as HTMLElement
    grid.innerHTML = ""
    allHeaders.forEach((header) => {
        var style = `"grid-row: ${header.row}; grid-column-start: ${header.columnStart}; grid-column-end: ${header.columnEnd}"`
        grid.innerHTML += `<span style=${style}>${header.value}</span>\n`
    })

    // generate row labels for all the seeds
    var allSeedsSet = new Set<number>()
    for (const iset of allImageSets.values()) {
        for (const img of iset.images) {
            allSeedsSet.add(img.seed)
        }
    }
    var allSeeds = Array.from(allSeedsSet).sort()
    for (const [idx, seed] of allSeeds.entries()) {
        var style = `"grid-row: ${idx + fields.length + 1}; grid-column: 1"`
        grid.innerHTML += `<span style=${style}>${seed}</span>`
    }

    // do the images!
    var imagesHTML = ""
    for (const [isetIdx, setKey] of allImageSetKeys.entries()) {
        var iset = allImageSets.get(setKey) as GImageSet
        var column = isetIdx + 2
        for (const [imgIdx, img] of iset.images.entries()) {
            var row = imgIdx + fields.length + 1
            
            style = `"grid-row: ${row}; grid-column: ${column}"`
            imagesHTML += `<span style=${style} class="image">\n`
            imagesHTML += `  <img src="${img.filename}" class="thumbnail"/>\n`
            imagesHTML += `  <span class="details">\n`
            imagesHTML += `    <img src="${img.filename}" class="fullsize"/>\n`
            imagesHTML += `    <div class="details_grid">\n`
            imagesHTML += `      <span class="detailsKey">model</span><span class="detailsVal">${iset.modelStr}</span>\n`
            imagesHTML += `      <span class="detailsKey">prompt</span><span class="detailsVal">"${iset.prompt}"</span>\n`
            imagesHTML += `      <span class="detailsKey">sampler</span><span class="detailsVal">${iset.sampler} ${iset.samplerSteps}</span>\n`
            imagesHTML += `      <span class="detailsKey">CFG</span><span class="detailsVal">${iset.cfg}</span>\n`
            imagesHTML += `      <span class="detailsKey">seed</span><span class="detailsVal">${img.seed}</span>\n`
            imagesHTML += `    </div>\n`
            imagesHTML += `  </span>\n`
            imagesHTML += "</span>\n"
        }
    }
    grid.innerHTML += imagesHTML
    return null;
}

update().then((val) => {
    console.log("done with updateList")
})
